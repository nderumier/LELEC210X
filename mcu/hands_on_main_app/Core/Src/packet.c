/*
 * packet.c
 */

#include "aes_ref.h"
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

void tag_cbc_mac(uint8_t *tag, const uint8_t *msg, size_t msg_len) {
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
//
//#include <string.h>
//
//extern CRYP_HandleTypeDef hcryp;
//
//#define PADDED_LEN 832
//
//static uint8_t iv[16] = {0};
//
//void tag_cbc_mac_dma(uint8_t *tag, const uint8_t *msg, size_t msg_len)
//{
//    static uint8_t buffer[PADDED_LEN] __attribute__((aligned(4)));
//    static uint8_t ciphertext[PADDED_LEN] __attribute__((aligned(4)));
//
//    size_t padded_len = (msg_len + 15) & ~0xF;   // round up to multiple of 16
//
//    /* Copy message */
//    memcpy(buffer, msg, msg_len);
//
//    /* Zero padding */
//    memset(buffer + msg_len, 0, padded_len - msg_len);
//
//    HAL_CRYP_SetInitVector(&hcryp, iv);
//
//    HAL_CRYP_AESCBC_Encrypt_DMA(&hcryp,
//                                buffer,
//                                padded_len,
//                                ciphertext);
//
//    /* Wait for completion */
//    while (HAL_CRYP_GetState(&hcryp) != HAL_CRYP_STATE_READY);
//
//    /* Last block = CBC-MAC */
//    memcpy(tag, &ciphertext[padded_len - 16], 16);
//}

//
//extern CRYP_HandleTypeDef hcryp;
//
//#define PADDED_LEN 832
//
//static uint8_t iv[16] = {0};
//
//static uint8_t buffer[PADDED_LEN] __attribute__((aligned(4)));
//static uint8_t ciphertext[PADDED_LEN] __attribute__((aligned(4)));
//
//volatile uint8_t cryp_done = 0;
//
//void HAL_CRYP_OutCpltCallback(CRYP_HandleTypeDef *hcryp)
//{
//    cryp_done = 1;
//}
//
//void tag_cbc_mac_it(uint8_t *tag, const uint8_t *msg, size_t msg_len)
//{
//    size_t padded_len = (msg_len + 15) & ~0xF;
//
//    cryp_done = 0;
//
//    memcpy(buffer, msg, msg_len);
//    memset(buffer + msg_len, 0, padded_len - msg_len);
//
//    HAL_CRYP_SetInitVector(&hcryp, iv);
//
//    HAL_CRYP_AESCBC_Encrypt_IT(&hcryp,
//                               buffer,
//                               padded_len,
//                               ciphertext);
//
//    while (!cryp_done);
//
//    memcpy(tag, &ciphertext[padded_len - 16], 16);
//}




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

    tag_cbc_mac(packet + PACKET_HEADER_LENGTH + payload_len,
                packet,
                PACKET_HEADER_LENGTH + payload_len);

//    tag_cbc_mac_it(packet + PACKET_HEADER_LENGTH + payload_len,
//            packet,
//            PACKET_HEADER_LENGTH + payload_len);

    return packet_len;
}
