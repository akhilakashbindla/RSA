import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange, getrandbits

# Function to power a number (for RSA algorithm)
def power(a, d, n):
    ans = 1
    while d != 0:
        if d % 2 == 1:
            ans = ((ans % n) * (a % n)) % n
        a = ((a % n) * (a % n)) % n
        d >>= 1
    return ans

# Miller-Rabin primality test function
def MillerRabin(N, d):
    a = randrange(2, N - 1)
    x = power(a, d, N)
    if x == 1 or x == N - 1:
        return True
    else:
        while d != N - 1:
            x = ((x % N) * (x % N)) % N
            if x == 1:
                return False
            if x == N - 1:
                return True
            d <<= 1
    return False

# Function to check if a number is prime
def is_prime(N, K):
    if N == 3 or N == 2:
        return True
    if N <= 1 or N % 2 == 0:
        return False

    # Find d such that d*(2^r) = N-1
    d = N - 1
    while d % 2 == 0:
        d //= 2

    for _ in range(K):
        if not MillerRabin(N, d):
            return False
    return True

# Function to generate a prime candidate
def generate_prime_candidate(length):
    p = getrandbits(length)
    p |= (1 << length - 1) | 1
    return p

# Function to generate a prime number of a specified bit length
def generatePrimeNumber(length):
    A = 4
    while not is_prime(A, 128):
        A = generate_prime_candidate(length)
    return A

# Generate two large prime numbers P and Q
length = 5
P = generatePrimeNumber(length)
Q = generatePrimeNumber(length)

print(P)
print(Q)

# Calculate N and Euler's Totient function
N = P * Q
eulerTotient = (P - 1) * (Q - 1)
print(N)
print(eulerTotient)

# Function to calculate GCD
def GCD(a, b):
    if a == 0:
        return b
    return GCD(b % a, a)

# Find E such that GCD(E, eulerTotient) = 1
E = generatePrimeNumber(4)
while GCD(E, eulerTotient) != 1:
    E = generatePrimeNumber(4)
print(E)

# Extended Euclidean Algorithm to find D
def gcdExtended(E, eulerTotient):
    a1, a2, b1, b2, d1, d2 = 1, 0, 0, 1, eulerTotient, E

    while d2 != 1:
        k = d1 // d2

        temp = a2
        a2 = a1 - (a2 * k)
        a1 = temp

        temp = b2
        b2 = b1 - (b2 * k)
        b1 = temp

        temp = d2
        d2 = d1 - (d2 * k)
        d1 = temp

        D = b2

    if D > eulerTotient:
        D = D % eulerTotient
    elif D < 0:
        D = D + eulerTotient

    return D

D = gcdExtended(E, eulerTotient)
print(D)

# Load the image
my_img = cv2.imread('c:/Users/akhil akash/RSA/Python/RSA.jpg')  # Use absolute path to your image
if my_img is None:
    raise ValueError("Image not found. Please check the path.")

# Encrypt the image
row, col = my_img.shape[0], my_img.shape[1]
enc = [[0 for x in range(col)] for y in range(row)]

for i in range(100, min(700, row)):
    for j in range(100, min(1000, col)):
        r, g, b = my_img[i, j]
        C1 = power(r, E, N)
        C2 = power(g, E, N)
        C3 = power(b, E, N)
        enc[i][j] = [C1, C2, C3]
        C1 = C1 % 256
        C2 = C2 % 256
        C3 = C3 % 256
        my_img[i, j] = [C1, C2, C3]

# Display the encrypted image
plt.imshow(cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB))
plt.show()

# Decrypt the image
for i in range(100, min(700, row)):
    for j in range(100, min(1000, col)):
        r, g, b = enc[i][j]
        M1 = power(r, D, N)
        M2 = power(g, D, N)
        M3 = power(b, D, N)
        my_img[i, j] = [M1, M2, M3]

# Display the decrypted image
plt.imshow(cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB))
plt.show()
