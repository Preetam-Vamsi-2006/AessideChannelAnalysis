#!/usr/bin/env python3
"""Quick test for the decryption functionality"""

import os
import sys
import base64
from Crypto.Cipher import AES

# Test data
test_plaintext = "Hello, this is a test message for AES encryption!"
test_key = os.urandom(16)
test_key_hex = test_key.hex().upper()

# Pad the text
def pad(text):
    encoded = text.encode('utf-8')
    remainder = len(encoded) % 16
    if remainder != 0:
        text += " " * (16 - remainder)
    return text

# Encrypt
cipher = AES.new(test_key, AES.MODE_ECB)
encrypted = cipher.encrypt(pad(test_plaintext).encode('utf-8'))
ciphertext_b64 = base64.b64encode(encrypted).decode()

print("Original Plaintext:", test_plaintext)
print("Key (Hex):", test_key_hex)
print("Ciphertext (Base64):", ciphertext_b64)
print()

# Decrypt
raw_key = bytes.fromhex(test_key_hex)
ciphertext_bytes = base64.b64decode(ciphertext_b64)
cipher2 = AES.new(raw_key, AES.MODE_ECB)
decrypted = cipher2.decrypt(ciphertext_bytes)
plaintext_recovered = decrypted.rstrip(b' ').decode('utf-8', errors='ignore')

print("Decrypted Plaintext:", plaintext_recovered)
print()
print("Match:", test_plaintext == plaintext_recovered)
