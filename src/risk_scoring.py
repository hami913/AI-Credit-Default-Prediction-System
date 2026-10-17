import math


def assess_risk(
    default_probability,
    low_cutoff=0.30,
    high_cutoff=0.60
):
    """
    Convert default probability into a 0-100 risk index.

    Higher score = Higher default risk.

    Thresholds are illustrative and require validation.
    """

    probability = float(default_probability)

    if not math.isfinite(probability):
        raise ValueError("Probability must be finite.")

    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    if not 0 <= low_cutoff < high_cutoff <= 1:
        raise ValueError("Invalid risk thresholds.")

    risk_index = round(probability * 100, 2)

    if probability < low_cutoff:
        risk_level = "Low"
    elif probability < high_cutoff:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "default_probability": probability,
        "risk_index": risk_index,
        "risk_level": risk_level
    }


if __name__ == "__main__":
    sample_probability = 0.5764544871918588
    result = assess_risk(sample_probability)

    print("Credit Risk Assessment")
    print("----------------------")
    print("Default Probability:",
          round(result["default_probability"] * 100, 2), "%")
    print("Risk Index:", result["risk_index"])
    print("Risk Level:", result["risk_level"])
