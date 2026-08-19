import numpy as np

def generate_report(feature_names, shap_vals, input_df):

    report = []

    for feature, val in zip(feature_names, shap_vals):

        val = float(np.array(val).flatten()[0])
        actual_value = input_df[feature].values[0]

        # NEGATIVE IMPACT
        if val < 0:

            if feature == "Outstanding_Debt":
                report.append(f"❌ High debt ({actual_value}) is lowering your score. Reduce it.")

            elif feature == "Num_of_Loan":
                report.append(f"❌ Too many loans ({actual_value}). Avoid new loans.")

            elif feature == "Total_EMI_per_month":
                report.append(f"❌ High EMI burden ({actual_value}). Try reducing EMIs.")

            elif feature == "Annual_Income":
                report.append(f"❌ Income ({actual_value}) is low compared to obligations.")

            elif feature == "Monthly_Balance":
                report.append(f"❌ Low balance ({actual_value}). Increase savings.")

            else:
                report.append(f"❌ {feature} is negatively impacting your score.")

        # POSITIVE IMPACT
        else:

            if feature == "Annual_Income":
                report.append(f"✅ Good income ({actual_value}) is helping your score.")

            elif feature == "Monthly_Balance":
                report.append(f"✅ Healthy balance ({actual_value}). Keep it up.")

            elif feature == "Outstanding_Debt":
                report.append(f"✅ Debt level is under control.")

            else:
                report.append(f"✅ {feature} is contributing positively.")

    # FINAL SUMMARY
    positives = sum(1 for v in shap_vals if float(np.array(v).flatten()[0]) > 0)
    negatives = len(shap_vals) - positives

    if negatives > positives:
        overall = "⚠ Your score can improve significantly with better financial habits."
    else:
        overall = "🎯 You are on the right track. Maintain consistency."

    return report, overall