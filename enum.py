from random import randint
from time import perf_counter
import numpy as np

#Gram-Schmidt直交化
def Gram_Schmidt(b):
  '''
  入力　b: n次元格子の基（行列）
  出力　GSOb: 入力基のGSOベクトル（行列）
        mu: 入力基のGSO係数（行列）
  '''
  m = b.shape[0]
  n = b.shape[1]
  GSOb = np.zeros((n, m))
  mu = np.identity(n)

  for i in range(n):
    GSOb[i] = b[i]
    for j in range(i):
      mu[i, j] = np.dot(b[i], GSOb[j]) / np.dot(GSOb[j], GSOb[j])
      GSOb[i] -= mu[i, j] * GSOb[j]
  
  return GSOb, mu

#二乗normの作成
def make_square_norm(GSOb):
  '''
  入力　GSOb:n次元格子の基（行列）
  出力　B:GSObの各横vectorの二乗normを並べたもの（vector）
  '''
  n = GSOb.shape[1]
  B = np.zeros(n)
  B = np.sum(np.abs(GSOb) ** 2, axis = 1)
  return B

#係数vectorを格子vectorに変換
def coef2lattice(v, b):
  n = b.shape[1]
  x = np.zeros(n)
  for i in range(n):
    x += v[i] * b[i]
  return x

#体積
def vol(B):
  GSOb, mu = Gram_Schmidt(B)
  return np.prod(GSOb)

#数え上げ
def ENUM(mu, B, R, b):
  '''
  入力　mu：GSO-vector（行列）
  　　　B：GSO-vectorの二乗norm
  　　　R：数え上げ上界列
  出力　短いvectorの係数vector
  '''
  n = len(B)
  sigma = np.zeros((n + 1, n))
  r = np.arange(n + 1); r -= 1; r = np.roll(r, -1)
  rho = np.zeros(n + 1)
  v = np.zeros(n); v[0] = 1
  c = np.zeros(n)
  w = np.zeros(n)
  last_nonzero = 0
  k = 0
  while 1:
    rho[k] = rho[k + 1] + (v[k] - c[k]) ** 2 * B[k]
    if rho[k] <= R[n - 1 - k] ** 2:
      if k == 0:
        return v
      k -= 1
      r[k - 1] = max(r[k - 1], r[k])
      for i in range(k + 1, r[k] + 1)[::-1]:
        sigma[i, k] = sigma[i + 1, k] + mu[i, k] * v[i]
      c[k] = -sigma[k + 1, k]
      v[k] = np.round(c[k])
      w[k] = 1
    else:
      k += 1
      if k == n:
        return np.empty(0)
      r[k - 1] = k
      if k >= last_nonzero:
        last_nonzero = k
        v[k] += 1
      else:
        if v[k] > c[k]:
          v[k] -= w[k]
        else:
          v[k] += w[k]
        w[k] += 1
    print("********\nv =",v,"\nvector =",coef2lattice(v, b),"\n||vector|| =",np.linalg.norm(coef2lattice(v, b)),"\n********\n")

#main
random = 0

n =10
eps = 0.999
B = np.zeros(n)
b = np.zeros((n, n))
ENUM_v = np.zeros(n)

if random:
  b = b.T; b[0] += np.random.randint(1, 1000, n); b = b.T
  for i in range(n):
    b[i, i] = randint(1, 1000)
else:
  b = np.array([[63, -14, -1, 84, 61], [74, -20, 23, -32, -52], [93, -46, -19, 0, -63], [93, 11, 13, 60, 52], [33, -93, 12, 57, -2]])#自分で設定してください

print("入力基行列：\n", b)
GSOb, mu = Gram_Schmidt(b)
B = make_square_norm(GSOb)

R = np.full(n, np.sqrt(n) * vol(b) ** (1 / n))

start_time = perf_counter()
ENUM_v = ENUM(mu, B, R, b)
end_time = perf_counter()

print("\n短いvectorの係数vector:\n", ENUM_v)
print("\n短いvector：\n", coef2lattice(ENUM_v, b))
print("\nRun time =", end_time - start_time,"secs")
