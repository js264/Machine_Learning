import random
from fractions import Fraction


def validate_priors(priors):
    total = sum(priors.values())
    if abs(total - 1) > 1e-9:
        raise ValueError(f"사전확률의 합이 1이 아닙니다: {total}")


def joint_probabilities(priors, likelihoods):
    """P(A_i ∩ B) = P(A_i) * P(B | A_i)"""
    return {k: priors[k] * likelihoods[k] for k in priors}


def total_probability(joint):
    """전체확률 P(B) = Σ P(A_i ∩ B)"""
    return sum(joint.values())


def posterior_probabilities(priors, likelihoods):
    """P(A_i | B) = P(A_i ∩ B) / P(B)"""
    validate_priors(priors)
    joint = joint_probabilities(priors, likelihoods)
    p_b = total_probability(joint)
    return {k: v / p_b for k, v in joint.items()}


def map_estimate(posteriors):
    """사후확률이 가장 큰 플랫폼 (MAP 추정)"""
    return max(posteriors, key=posteriors.get)


def main():
    priors = {"PC": 0.1, "모바일 앱": 0.6, "모바일 웹": 0.3}
    likelihoods = {"PC": 0.8, "모바일 앱": 0.2, "모바일 웹": 0.5}

    joint = joint_probabilities(priors, likelihoods)
    p_b = total_probability(joint)
    posteriors = posterior_probabilities(priors, likelihoods)

    print("=" * 62)
    print(" 베이즈 정리: P(플랫폼 | 구매)")
    print("=" * 62)
    print(f"{'플랫폼':<8}{'P(A)':>10}{'P(B|A)':>10}{'P(A∩B)':>10}{'P(A|B)':>12}{'분수':>8}")
    print("-" * 62)
    for k in priors:
        # 분수 표현: 소수 오차를 없애기 위해 문자열에서 Fraction 생성
        frac = Fraction(str(priors[k])) * Fraction(str(likelihoods[k])) / Fraction(str(round(p_b, 10)))
        print(f"{k:<8}{priors[k]:>10.2f}{likelihoods[k]:>10.2f}{joint[k]:>10.4f}"
              f"{posteriors[k]:>12.4f}{str(frac):>8}")
    print("-" * 62)
    print(f"전체 구매 확률 P(B) = {p_b:.4f}")
    print(f"사후확률 합계       = {sum(posteriors.values()):.4f}")

    best = map_estimate(posteriors)
    print(f"\n▶ 결론: 구매자는 '{best}' 사용자였을 확률이 가장 높다 (약 {posteriors[best]:.2%})")


if __name__ == "__main__":
    main()
