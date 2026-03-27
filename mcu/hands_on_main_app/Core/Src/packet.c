/*
 * packet.c
 */

#include "aes_ref.h"
#include "aes.h"
#include "config.h"
#include "packet.h"
#include "main.h"
#include "utils.h"
#include <string.h>

const uint8_t AES_Key[16]  = {
                            0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00};

static volatile uint8_t cryp_done = 0;
static volatile uint8_t cryp_error = 0;
static uint8_t cryp_input_buf[PACKET_LENGTH + 16];
static uint8_t cryp_output_buf[PACKET_LENGTH + 16];

void HAL_CRYP_OutCpltCallback(CRYP_HandleTypeDef *hcryp_cb)
{
    if (hcryp_cb->Instance == AES) {
        cryp_done = 1;
    }
}

void HAL_CRYP_ErrorCallback(CRYP_HandleTypeDef *hcryp_cb)
{
    if (hcryp_cb->Instance == AES) {
        cryp_error = 1;
        cryp_done = 1;
    }
}

static int tag_cbc_mac_hw(uint8_t *tag, const uint8_t *msg, size_t msg_len)
{
    if (msg_len > PACKET_LENGTH) {
        return -1;
    }

    size_t padded_len = ((msg_len + 15U) / 16U) * 16U;
    if (padded_len == 0U) {
        padded_len = 16U;
    }

    memcpy(cryp_input_buf, msg, msg_len);
    memset(cryp_input_buf + msg_len, 0, padded_len - msg_len);

    if (HAL_CRYP_DeInit(&hcryp) != HAL_OK) {
        return -1;
    }
    if (HAL_CRYP_Init(&hcryp) != HAL_OK) {
        return -1;
    }

    cryp_done = 0;
    cryp_error = 0;

    if (HAL_CRYP_AESCBC_Encrypt_IT(&hcryp,
                                   (uint32_t *)cryp_input_buf,
                                   (uint16_t)padded_len,
                                   (uint32_t *)cryp_output_buf) != HAL_OK) {
        return -1;
    }

    uint32_t t0 = HAL_GetTick();
    while (cryp_done == 0U) {
        if ((HAL_GetTick() - t0) > 100U) {
            return -1;
        }
    }

    if (cryp_error != 0U) {
        return -1;
    }

    memcpy(tag, cryp_output_buf + padded_len - 16U, 16U);
    return 0;
}

static void tag_cbc_mac_sw(uint8_t *tag, const uint8_t *msg, size_t msg_len) {
    // statew is used as the 16-byte AES state (s in the algorithm).
    // It stays constant size regardless of msg_len.
    uint32_t statew[4] = {0};
    uint8_t *state = (uint8_t*) statew;  // s ← 0^16

    size_t offset = 0;

    // Process full 16-byte blocks first
    while (offset + 16 <= msg_len) {
        // s ← s ⊕ xi
        for (int i = 0; i < 16; i++) {
            state[i] ^= msg[offset + i];
        }

        // s ← AES_k(s)
        AES128_encrypt(state, AES_Key);

        offset += 16;
    }

    // Handle last block (possibly partial)
    uint8_t last_block[16];   // temporary 16-byte block on the stack

    size_t rem = msg_len - offset;

    // Copy the remaining bytes (possibly 0 bytes if exactly aligned)
    for (size_t i = 0; i < rem; i++) {
        last_block[i] = msg[offset + i];
    }
    // Zero padding for the rest
    for (size_t i = rem; i < 16; i++) {
        last_block[i] = 0x00;
    }

    // Process last block: s ← AES_k(s ⊕ x_n)
    for (int i = 0; i < 16; i++) {
        state[i] ^= last_block[i];
    }

    AES128_encrypt(state, AES_Key);

    // Output tag = s
    for (int j = 0; j < 16; j++) {
        tag[j] = state[j];
    }
}

// Assumes payload is already in place in the packet
int make_packet(uint8_t *packet, size_t payload_len, uint8_t sender_id, uint32_t serial) {

    size_t packet_len = payload_len + PACKET_HEADER_LENGTH + PACKET_TAG_LENGTH;

    // ---------------------------------------------------------------------
    // Construct the packet header (big-endian)
    // ---------------------------------------------------------------------

    // r (reserved)
    packet[0] = 0x00;

    // emitter_id
    packet[1] = sender_id;

    // payload_length (2 bytes, BE)
    packet[2] = (payload_len >> 8) & 0xFF;   // high byte
    packet[3] =  payload_len       & 0xFF;   // low byte

    // packet_serial (4 bytes, BE)
    packet[4] = (serial >> 24) & 0xFF;
    packet[5] = (serial >> 16) & 0xFF;
    packet[6] = (serial >>  8) & 0xFF;
    packet[7] =  serial        & 0xFF;

    // ---------------------------------------------------------------------
    // Tag field (already incorrectly filled by tag_cbc_mac, as instructed)
    // ---------------------------------------------------------------------

    // Software AES-CBC-MAC (legacy/reference path)
    tag_cbc_mac_sw(packet + PACKET_HEADER_LENGTH + payload_len,
                   packet,
                   PACKET_HEADER_LENGTH + payload_len);

    // Hardware AES-CBC-MAC (interrupt path)
//     tag_cbc_mac_hw(packet + PACKET_HEADER_LENGTH + payload_len,
//                          packet,
//                          PACKET_HEADER_LENGTH + payload_len);

    return packet_len;
}
