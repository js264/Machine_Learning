import math

EPS = 1e-9


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
    (a, b), (c, d) = A
    # 첫 행 (a−λ)x + by = 0 → v = (b, λ−a)
    if abs(b) > EPS or abs(a - lam) > EPS:
        return [b, lam - a]
    # 둘째 행 cx + (d−λ)y = 0 → v = (λ−d, c)
    if abs(c) > EPS or abs(d - lam) > EPS:
        return [lam - d, c]
    return [1.0, 0.0]  # A − λI = 0 이면 모든 벡터가 고유벡터



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
    return sum(a * b for a, b in zip(u, v))



# ===== [4] 정렬 변환 행렬 =====

def transpose(M):  # ppt 12페이지 참고
    """전치 (M^T)_ij = M_ji"""
    return [list(row) for row in zip(*M)]



def alignment_matrix(v1, v2):  # ppt 14, 37페이지 참고
    """고유벡터를 열로 쌓은 Q 를 만들고, x·y축 정렬 변환 T = Q^-1 = Q^T 반환"""
    Q = transpose([v1, v2])
    if det_2x2(Q) < 0:  # 반사가 아닌 회전이 되도록 v2 부호 반전
        v2 = [-x for x in v2]
        Q = transpose([v1, v2])
    return transpose(Q)



# ===== [5] 변환 종류 + 회전 각도 =====

def is_identity(M):
    return all(math.isclose(M[i][j], 1.0 if i == j else 0.0, abs_tol=EPS)
               for i in range(len(M)) for j in range(len(M)))



def classify(T):  # ppt 30~34페이지 참고
    """변환 종류 판별: 회전 / 크기 변환 / 전단 변환"""
    off = [T[0][1], T[1][0]]
    if is_identity(mat_mul(transpose(T), T)) and math.isclose(det_2x2(T), 1.0, abs_tol=EPS):
        return "회전 (Rotation)", math.degrees(math.atan2(T[1][0], T[0][0]))
    if all(abs(x) < EPS for x in off):
        return "크기 변환 (Scale)", None
    if (math.isclose(T[0][0], 1.0, abs_tol=EPS) and math.isclose(T[1][1], 1.0, abs_tol=EPS)
            and sum(abs(x) > EPS for x in off) == 1):
        return "전단 변환 (Shear)", None
    return "기타 선형 변환", None



# ===== [6] 정렬된 새 행렬 T·A·T^T =====

def mat_mul(A, B):  # ppt 13페이지 참고
    """행렬 곱 c_ij = Σ a_ik * b_kj"""
    if len(A[0]) != len(B):
        raise ValueError("안쪽 차원이 같지 않아 곱할 수 없습니다.")
    Bt = transpose(B)
    return [[dot(row, col) for col in Bt] for row in A]



# ===== 출력 보조 =====

def fmt_matrix(M, indent="    "):
    return "\n".join(indent + "[" + ", ".join(f"{x:8.4f}" for x in row) + "]" for row in M)



def fmt_vector(v):
    return "(" + ", ".join(f"{x:.4f}" for x in v) + ")"



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
    for lam, v in ((lam1, v1), (lam2, v2)):
        Av = mat_mul(A, [[x] for x in v])
        print(f"    검산 Av = {fmt_vector([r[0] for r in Av])} = {lam:.0f}·v")

    # 3. 직교 확인
    d = dot(v1, v2)
    print("\n[3] 직교 확인")
    print(f"    v1·v2 = {d:.4f} → {'수직(직교)' if math.isclose(d, 0, abs_tol=EPS) else '수직 아님'}")

    # 4. 정렬 변환 행렬
    T = alignment_matrix(v1, v2)
    print("\n[4] 정렬 변환 행렬 T = Q^T (Q = [v1 v2], 직교행렬이라 Q^-1 = Q^T)")
    print(fmt_matrix(T))

    # 5. 변환 종류 + 회전 각도
    kind, angle = classify(T)
    print("\n[5] 변환 종류")
    print(f"    {kind}, det(T) = {det_2x2(T):.4f}, T^T·T = I")
    if angle is not None:
        print(f"    회전 각도 θ = {angle:.2f}° ({'반시계' if angle > 0 else '시계'} 방향)")

    # 6. 정렬된 새 행렬
    A_aligned = mat_mul(mat_mul(T, A), transpose(T))
    print("\n[6] 정렬된 행렬 T·A·T^T")
    print(fmt_matrix([[0.0 if abs(x) < EPS else x for x in row] for row in A_aligned]))
    print(f"    → 대각 성분 = 고윳값 ({lam1:.0f}, {lam2:.0f}), 비대각 성분 = 0")

    print(f"\n▶ 결론: T({angle:.2f}° 회전)로 정렬하면 A는 x축 방향 {lam1:.0f}배, y축 방향 {lam2:.0f}배로 늘이는 행렬이 된다")



if __name__ == "__main__":
    main()
