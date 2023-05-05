from random import randint
from time import perf_counter
import numpy as np

#Gram-Schmidt直交化
def Gram_Schmidt(b):
  '''
  入力　b: n次元格子の基行列
  出力　GSOb: 入力基のGram-Schmidt直交化vector
        mu: 入力基のGram-Schmidt直交化係数行列
  '''
  m = b.shape[0]
  n = b.shape[1]
  GSOb = np.zeros((n, m))
  mu = np.identity(n)

  for i in range(n):
    GSOb[i] = b[i].copy()
    for j in range(i):
      mu[i, j] = np.dot(b[i], GSOb[j]) / np.dot(GSOb[j], GSOb[j])
      GSOb[i] -= mu[i, j] * GSOb[j].copy()
  
  return GSOb, mu

#二乗normの作成
def make_square_norm(GSOb):
  '''
  入力　GSOb:n次元格子の基行列
  出力　B:GSObの各横vectorの二乗normを並べたもの（vector）
  '''
  n = GSOb.shape[1]
  B = np.zeros(n)
  B = np.sum(np.abs(GSOb) ** 2, axis = 1)
  return B

#係数vectorを格子vectorに変換
def coef2lattice(v, b):
  '''
  入力　v:係数vector
  　　　b:格子の基行列
  出力　vに対応する格子vector
  '''
  n = b.shape[1]
  x = np.zeros(n, int)
  for i in range(n):
    x += v[i] * b[i].copy()
  return x

#体積
def vol(B):
  '''
  入力　B：n次元格子Lの基行列
  出力　Lの体積
  '''
  GSOb, mu = Gram_Schmidt(B)
  return np.prod(GSOb)

#@title 数え上げ
def ENUM(mu, B, R, b):
  '''
  入力　mu：Gram-Schmidt直交化係数行列
  　　　B：GSO-vectorの二乗normを並べたもの(vector)
  　　　R：数え上げ上界(float)
  出力　短いvectorの係数vector
  '''
  n = B.size
  sigma = np.zeros((n + 1, n))
  r = np.arange(n + 1); r -= 1; r = np.roll(r, -1)
  rho = np.zeros(n + 1)
  v = np.zeros(n, int); v[0] = 1
  c = np.zeros(n)
  w = np.zeros(n, int)
  last_nonzero = 0
  k = 0
  counter = 0
  while 1:
    counter += 1
    rho[k] = rho[k + 1] + (v[k] - c[k]) ** 2 * B[k]
    if rho[k] <= R ** 2:
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
    #print("********\nv =",v,"\nvector =",coef2lattice(v, b),"\n||vector|| =",np.linalg.norm(coef2lattice(v, b)),"\n********\n")

#main
def main():
  random = True #randomに行列を生成するか
  n =10 #格子次元
  eps = 0.999
  B = np.zeros(n)
  b = np.zeros((n, n), int)
  ENUM_v = np.zeros(n)

  if random:
    while 1:
      b = np.random.randint(1, 1000, (n, n))
      if np.linalg.det(b) != 0:
        break
  else:
    b = np.array([[63, -14, -1, 84, 61], [74, -20, 23, -32, -52], [93, -46, -19, 0, -63], [93, 11, 13, 60, 52], [33, -93, 12, 57, -2]])#自分で設定してください
    #b = np.array([[-79, 35, 31, 83, -66, 35, -32, 46, 21, 2], [43, -64, -37, -31, -27, -7, -42, 21, 16, 16], [-1, -97, -91, -43, 19, -21, -65, -36, 34, -55], [-58, -38, 87, 42, 94, -83, 66, -69, -2, -30], [84, -61, 93, -67, 3, 94, 31, 27, -60, 98], [-1, 34, 58, -38, 29, 67, -18, 15, -75, -16], [19, 16, 52, 32, -20, 55, 94, -34, 4, 80], [-58, -17, 99, 93, -49, -53, 24, 51, 5, 93], [17, 31, 78, 53, 40, -22, -39, 7, 70, -98], [93, -6, -7, -12, 79, -40, 27, -95, 98, 20]])

  print("入力基行列：\n", b)
  GSOb, mu = Gram_Schmidt(b)
  B = make_square_norm(GSOb)

  R =  np.sqrt(n) * vol(b) ** (1 / n) #数え上げ上界、初期値はMinkowski's theoremに基づく上界

  start_time = perf_counter()
  ENUM_v = ENUM(mu, B, R, b)
  end_time = perf_counter()

  print("\n短いvectorの係数vector:\n", ENUM_v)
  print("\n短いvector：\n", coef2lattice(ENUM_v, b))
  print("\nRun time =", end_time - start_time,"secs")

main()
