def validate_priors(priors):
    total = sum(priors.values())
    if total != 1:
        raise ValueError(f"사전확률의 합이 1이 아닙니다: {total}")


def joint_probabilities(priors, likelihoods):  # ppt 10페이지 참고
    """곱셈법칙 P(A ∩ B) =  P(A | B) * P(B) = P(A) * P(B | A) """
    return {k: priors[k] * likelihoods[k] for k in priors}


def total_probability(joint):  # ppt 14페이지 참고
    """전체확률 P(B) = Σ P(B | A) * P(A) = Σ P(A ∩ B)"""
    return sum(joint.values())


def posterior_probabilities(priors, likelihoods):  # ppt 11, 15페이지 참고
    """베이즈 정리 P(A | B) = P(A ∩ B) / P(B)"""
    validate_priors(priors)
    joint = joint_probabilities(priors, likelihoods)
    p_b = total_probability(joint)
    return {k: v / p_b for k, v in joint.items()}


def max_estimate(posteriors):
    """사후확률이 가장 큰 플랫폼"""
    return max(posteriors, key=posteriors.get)


def main():
    priors = {"PC": 0.1, "App": 0.6, "Web": 0.3}
    likelihoods = {"PC": 0.8, "App": 0.2, "Web": 0.5}

    joint = joint_probabilities(priors, likelihoods)
    p_b = total_probability(joint)
    posteriors = posterior_probabilities(priors, likelihoods)

    print("=" * 50)
    print(" 베이즈 정리: P(플랫폼 | 구매)")
    print("=" * 50)
    print(f"{'Platform':<8}{'P(A)':>10}{'P(B|A)':>10}{'P(A∩B)':>10}{'P(A|B)':>12}")
    print("-" * 50)
    for k in priors:
        print(f"{k:<8}{priors[k]:>10.2f}{likelihoods[k]:>10.2f}{joint[k]:>10.4f}{posteriors[k]:>12.4f}")
    print("-" * 50)
    print(f"전체 구매 확률 P(B) = {p_b:.4f}")
    print(f"사후확률 합계       = {sum(posteriors.values()):.4f}")

    best = max_estimate(posteriors)
    print(f"\n▶ 결론: 구매자는 '{best}' 사용자였을 확률이 가장 높다 (약 {posteriors[best]:.2%})")


if __name__ == "__main__":
    main()