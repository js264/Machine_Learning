import math


# ===== [1] 고윳값 =====

def det_2x2(M):  # ppt 15페이지 참고
    """행렬식 det = ad − bc"""
    return M[0][0] * M[1][1] - M[0][1] * M[1][0]



def eigenvalues_2x2(A):  # ppt 40~41페이지 참고
    """특성방정식 det(A − λI) = λ² − tr(A)·λ + det(A) = 0 을 근의 공식으로 풀기"""
    tr = A[0][0] + A[1][1]
    disc = tr ** 2 - 4 * det_2x2(A)
    if disc < 0:
        raise ValueError("실수 고윳값이 존재하지 않습니다.")
    root = math.sqrt(disc)
    return [(tr - root) / 2, (tr + root) / 2]



# ===== [2] 고유벡터 + 정규화 =====

def eigenvector_2x2(A, lam):  # ppt 42페이지 참고
    """(A − λI)v = 0 의 해 중 0이 아닌 벡터"""
    a, b = A[0]
    # 첫 행 (a−λ)x + by = 0 → v = (b, λ−a)
    return [b, lam - a]



def norm(v):  # ppt 20페이지 참고
    """L2 노름 ‖v‖ = √(Σ v_i²)"""
    return math.sqrt(dot(v, v))



def normalize(v):
    """길이가 1이 되도록 정규화 v / ‖v‖"""
    n = norm(v)
    if n == 0:
        raise ValueError("영벡터는 정규화할 수 없습니다.")
    return [x / n for x in v]



# ===== [3] 직교 확인 =====

def dot(u, v):  # ppt 24~25페이지 참고
    """내적 u·v = Σ u_i * v_i"""
    total = 0
    for i in range(len(u)):
        total += u[i] * v[i]
    return total



# ===== [4] 정렬 변환 행렬 =====

def transpose(M):  # ppt 12페이지 참고
    """전치 (M^T)_ij = M_ji"""
    rows, cols = len(M), len(M[0])
    result = [[0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            result[j][i] = M[i][j]
    return result



def alignment_matrix(v1, v2):  # ppt 14, 37페이지 참고
    """고유벡터를 열로 쌓은 Q 를 만들고, x·y축 정렬 변환 T = Q^-1 = Q^T 반환"""
    Q = [[v1[0], v2[0]],
         [v1[1], v2[1]]]  # 1열 = v1, 2열 = v2
    if det_2x2(Q) < 0:  # 반사가 아닌 회전이 되도록 v2(2열) 부호 반전
        Q[0][1] = -Q[0][1]
        Q[1][1] = -Q[1][1]
    return transpose(Q)  # 직교행렬이라 Q^-1 = Q^T



# ===== [5] 변환 종류 + 회전 각도 =====

def classify(T):  # ppt 30~34페이지 참고
    """변환 종류 판별: 회전 / 크기 변환 / 전단 변환"""
    # 계산 결과가 0.9999999999999999 처럼 나올 수 있어서 소수점 6자리로 반올림해서 비교
    (a, b), (c, d) = [[round(x, 6) for x in row] for row in T]
    det = round(det_2x2(T), 6)
    if a == d and b == -c and det == 1:  # 회전 행렬 [[cosθ, −sinθ], [sinθ, cosθ]] 꼴
        angle = math.degrees(math.atan2(T[1][0], T[0][0]))  # tanθ = sinθ / cosθ
        return "회전 (Rotation)", angle
    if b == 0 and c == 0:  # 비대각 성분이 0
        return "크기 변환 (Scale)", None
    if a == 1 and d == 1 and (b == 0 or c == 0):  # 대각 1, 한쪽만 m
        return "전단 변환 (Shear)", None
    return "기타 선형 변환", None



# ===== [6] 정렬된 새 행렬 T·A·T^T =====

def mat_mul(A, B):  # ppt 13페이지 참고
    """행렬 곱 c_ij = Σ a_ik * b_kj"""
    if len(A[0]) != len(B):
        raise ValueError("안쪽 차원이 같지 않아 곱할 수 없습니다.")
    n, m, s = len(A), len(B[0]), len(B)
    C = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(s):
                C[i][j] += A[i][k] * B[k][j]
    return C



# ===== 출력 보조 =====

def fmt_matrix(M, indent="    "):
    """행렬을 한 행씩 줄바꿈해서 소수점 4자리로 정렬한 문자열"""
    lines = []
    for row in M:
        cells = [f"{x:8.4f}" for x in row]
        lines.append(indent + "[" + ", ".join(cells) + "]")
    return "\n".join(lines)



def fmt_vector(v):
    """벡터를 (x, y) 꼴의 소수점 4자리 문자열로"""
    cells = [f"{x:.4f}" for x in v]
    return "(" + ", ".join(cells) + ")"



def main():
    A = [[6, 2], [2, 9]]

    print("=" * 50)
    print(" 행렬을 x, y축에 정렬하는 변환 찾기")
    print("=" * 50)
    print("A =")
    print(fmt_matrix(A))

    # 1. 고윳값
    lam1, lam2 = eigenvalues_2x2(A)
    print("\n[1] 고윳값: det(A − λI) = 0")
    print(f"    λ² − {A[0][0] + A[1][1]}λ + {det_2x2(A)} = 0")
    print(f"    λ1 = {lam1:.4f}, λ2 = {lam2:.4f}")

    # 2. 고유벡터 + 정규화
    v1 = normalize(eigenvector_2x2(A, lam1))
    v2 = normalize(eigenvector_2x2(A, lam2))
    print("\n[2] 정규화된 고유벡터")
    print(f"    λ1 = {lam1:.0f} → v1 = {fmt_vector(v1)}, ‖v1‖ = {norm(v1):.4f}")
    print(f"    λ2 = {lam2:.0f} → v2 = {fmt_vector(v2)}, ‖v2‖ = {norm(v2):.4f}")
    # 검산: Av 의 각 원소 = A의 행과 v의 내적 (ppt 11페이지)
    Av1 = [dot(A[0], v1), dot(A[1], v1)]
    Av2 = [dot(A[0], v2), dot(A[1], v2)]
    print(f"    검산 Av1 = {fmt_vector(Av1)} = {lam1:.0f}·v1")
    print(f"    검산 Av2 = {fmt_vector(Av2)} = {lam2:.0f}·v2")

    # 3. 직교 확인
    d = dot(v1, v2)
    if round(d, 6) == 0:
        result = "수직(직교)"
    else:
        result = "수직 아님"
    print("\n[3] 직교 확인")
    print(f"    v1·v2 = {d:.4f} → {result}")

    # 4. 정렬 변환 행렬
    T = alignment_matrix(v1, v2)
    print("\n[4] 정렬 변환 행렬 T = Q^T (Q = [v1 v2], 직교행렬이라 Q^-1 = Q^T)")
    print(fmt_matrix(T))

    # 5. 변환 종류 + 회전 각도
    kind, angle = classify(T)
    print("\n[5] 변환 종류")
    print(f"    {kind}, det(T) = {det_2x2(T):.4f}, T^T·T = I")
    if angle is not None:
        if angle > 0:
            direction = "반시계"
        else:
            direction = "시계"
        print(f"    회전 각도 θ = {angle:.2f}° ({direction} 방향)")

    # 6. 정렬된 새 행렬
    TA = mat_mul(T, A)
    A_aligned = mat_mul(TA, transpose(T))
    print("\n[6] 정렬된 행렬 T·A·T^T")
    print(fmt_matrix(A_aligned))
    print(f"    → 대각 성분 = 고윳값 ({lam1:.0f}, {lam2:.0f}), 비대각 성분 = 0")

    print(f"\n▶ 결론: T({angle:.2f}° 회전)로 정렬하면 A는 x축 방향 {lam1:.0f}배, y축 방향 {lam2:.0f}배로 늘이는 행렬이 된다")



if __name__ == "__main__":
    main()
