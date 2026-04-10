#include <adc_dblbuf.h>
#include "config.h"
#include "main.h"
#include "spectrogram.h"
#include "arm_math.h"
#include "utils.h"
#include "s2lp.h"
#include "packet.h"

static volatile uint16_t ADCDoubleBuf[2*ADC_BUF_SIZE]; /* ADC group regular conversion data (array of data) */
static volatile uint16_t* ADCData[2] = {&ADCDoubleBuf[0], &ADCDoubleBuf[ADC_BUF_SIZE]};
static volatile uint8_t ADCDataRdy[2] = {0, 0};

static volatile uint8_t cur_melvec = 0;
static q15_t mel_vectors[N_MELVECS][MELVEC_LENGTH];

static uint32_t packet_cnt = 0;

static volatile int32_t rem_n_bufs = 0;

/* =========================
 * Adaptive noise estimation
 * =========================
 *
 * We do NOT use a fixed threshold.
 * We estimate the noise floor (energy) online with an EWMA.
 * Then we detect an event if E > alpha * noise_est.
 *
 * Tuning:
 * - NOISE_INIT_BUFS: number of buffers used to bootstrap the noise estimate
 * - NOISE_BETA_SHIFT: beta = 1/2^shift (how fast noise estimate tracks changes)
 * - ALPHA: detection aggressiveness (larger -> less sensitive)
 */
#define NOISE_INIT_BUFS     (50u)
#define NOISE_BETA_SHIFT    (6u)    /* beta = 1/64 */
#define NOISE_ALPHA_NUM     (8u)    /* alpha = 8 */
#define NOISE_ALPHA_DEN     (1u)

static uint32_t noise_est = 0;
static uint32_t noise_boot_sum = 0;
static uint32_t noise_boot_cnt = 0;
static uint8_t  noise_ready = 0;

static volatile uint8_t recording = 0;

/* Compute buffer energy (sum of squared centered samples) */
static inline uint32_t compute_energy_u16(const uint16_t *buf, uint32_t n)
{
	uint32_t e = 0;
	for (uint32_t i = 0; i < n; i++) {
		int32_t s = (int32_t)buf[i] - 2048;  /* 12-bit mid-scale */
		e += (uint32_t)(s * s);
	}
	return e;
}

/* Update noise estimate */
static inline void update_noise(uint32_t e)
{
	if (!noise_ready) {
		/* Bootstrap: average of first NOISE_INIT_BUFS energies */
		noise_boot_sum += e;
		noise_boot_cnt++;
		if (noise_boot_cnt >= NOISE_INIT_BUFS) {
			noise_est = noise_boot_sum / NOISE_INIT_BUFS;
			noise_ready = 1;
		}
		return;
	}

	/* EWMA update: N = N + beta*(E - N), beta=1/2^NOISE_BETA_SHIFT */
	int32_t diff = (int32_t)e - (int32_t)noise_est;
	noise_est = (uint32_t)((int32_t)noise_est + (diff >> NOISE_BETA_SHIFT));
}

/* Decide if sound is present (adaptive threshold) */
static inline uint8_t sound_detected_adaptive(uint32_t e)
{
	if (!noise_ready) return 0; /* don't trigger during learning phase */
	return (e > (NOISE_ALPHA_NUM * noise_est) / NOISE_ALPHA_DEN);
}

int StartADCAcq(int32_t n_bufs) {
	rem_n_bufs = n_bufs;
	cur_melvec = 0;
	if (rem_n_bufs != 0) {
		return HAL_ADC_Start_DMA(&hadc1, (uint32_t *)ADCDoubleBuf, 2*ADC_BUF_SIZE);
	} else {
		return HAL_OK;
	}
}

int IsADCFinished(void) {
	return (rem_n_bufs == 0);
}

static void StopADCAcq() {
	HAL_ADC_Stop_DMA(&hadc1);
}

static void print_spectrogram(void) {
#if (DEBUGP == 1)
	start_cycle_count();
	DEBUG_PRINT("Acquisition complete, sending the following FVs\r\n");
	for(unsigned int j=0; j < N_MELVECS; j++) {
		DEBUG_PRINT("FV #%u:\t", j+1);
		for(unsigned int i=0; i < MELVEC_LENGTH; i++) {
			DEBUG_PRINT("%.2f, ", q15_to_float(mel_vectors[j][i]));
		}
		DEBUG_PRINT("\r\n");
	}
	stop_cycle_count("Print FV");
#endif
}

static void print_encoded_packet(uint8_t *packet) {
#if (DEBUGP == 1)
	char hex_encoded_packet[2*PACKET_LENGTH+1];
	hex_encode(hex_encoded_packet, packet, PACKET_LENGTH);
	DEBUG_PRINT("DF:HEX:%s\r\n", hex_encoded_packet);
#endif
}

static void encode_packet(uint8_t *packet, uint32_t* packet_cnt) {
	// BE encoding of each mel coef
	for (size_t i=0; i<N_MELVECS; i++) {
		for (size_t j=0; j<MELVEC_LENGTH; j++) {
			(packet+PACKET_HEADER_LENGTH)[(i*MELVEC_LENGTH+j)*2]   = mel_vectors[i][j] >> 8;
			(packet+PACKET_HEADER_LENGTH)[(i*MELVEC_LENGTH+j)*2+1] = mel_vectors[i][j] & 0xFF;
		}
	}
	// Write header and tag into the packet.
	make_packet(packet, PAYLOAD_LENGTH, 0, *packet_cnt);
	*packet_cnt += 1;
	if (*packet_cnt == 0) {
		// Should not happen as packet_cnt is 32-bit and we send at most 1 packet per second.
		DEBUG_PRINT("Packet counter overflow.\r\n");
		Error_Handler();
	}
}

static void send_spectrogram() {
	uint8_t packet[PACKET_LENGTH];

	start_cycle_count();
	encode_packet(packet, &packet_cnt);
	stop_cycle_count("Encode packet");

	start_cycle_count();
	S2LP_Send(packet, PACKET_LENGTH);
	stop_cycle_count("Send packet");

//	print_encoded_packet(packet);
}

static void ADC_Callback(int buf_cplt) {
	/* In continuous mode (rem_n_bufs = -1), we do not decrement */
	if (rem_n_bufs != -1) {
		rem_n_bufs--;
	}

	if (rem_n_bufs == 0) {
		StopADCAcq();
	} else if (ADCDataRdy[1-buf_cplt]) {
		DEBUG_PRINT("Error: ADC Data buffer full\r\n");
		Error_Handler();
	}
	ADCDataRdy[buf_cplt] = 1;

	/* ===== 1) Adaptive noise tracking + detection on raw buffer energy ===== */
	const uint16_t *raw = (const uint16_t *)ADCData[buf_cplt];
	uint32_t E = compute_energy_u16(raw, ADC_BUF_SIZE);

	/* Noise should be updated mainly when not recording an event */
	if (!recording) {
		update_noise(E);
	}

	if (!recording) {
		/* If no event detected -> skip heavy processing */
		if (!sound_detected_adaptive(E)) {
			ADCDataRdy[buf_cplt] = 0;
			return;
		}

		/* Event starts: reset mel vector index */
		recording = 1;
		cur_melvec = 0;
	}

	/* ===== 2) Heavy processing only during an event ===== */
	Spectrogram_Format((q15_t *)ADCData[buf_cplt]);
	Spectrogram_Compute((q15_t *)ADCData[buf_cplt], mel_vectors[cur_melvec]);
	printf("DF:HEX:");
	for (int i = 0; i < MELVEC_LENGTH; i++) {
	    // Print each 16-bit Mel bin as 4 hex characters
	    // %04x ensures leading zeros are kept (e.g., 5 becomes 0005)
	    printf("%04x", (uint16_t)mel_vectors[cur_melvec][i]);
	}
	printf("\r\n");
	cur_melvec++;

	ADCDataRdy[buf_cplt] = 0;

	/* ===== 3) Once we have N_MELVECS -> send one packet and go back to listening ===== */
	if (recording && (cur_melvec >= N_MELVECS)) {
//		print_spectrogram();
		send_spectrogram();

		recording = 0;
		cur_melvec = 0;
	}
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
	ADC_Callback(1);
}

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc)
{
	ADC_Callback(0);
}
