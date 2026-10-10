# Research report: How do federated gradient boosting methods trade off privacy, accuracy and communication cost?

*Generated 2026-10-10 11:43 · run `20261010-114349-how-do-federated-gradient-boosting-metho` · 9 papers found, 5 read, 5 cited · 12/12 extracted claims verified · 1 search iteration(s)*

## Executive summary

Differentially private federated boosting is viable for predictive maintenance at moderate privacy budgets, but the accuracy cost at strong budgets and… [1] Quantisation cuts communication by 75 percent with less than 0.5 percent accuracy loss, and its rounding noise interacts with differential privacy noise in… [2] Communication cost grows linearly with the number of bins, and using 32 bins instead of 256 reduced traffic by 87 percent with an accuracy drop of only 0.3… [3]

## Findings

### SQ1: Which methods and system designs address how federated gradient boosting methods trade off privacy, accuracy and communication cost?

- Differentially private federated boosting is viable for predictive maintenance at moderate privacy budgets, but the accuracy cost at strong budgets and non-IID data needs further study. [1]
- Quantisation cuts communication by 75 percent with less than 0.5 percent accuracy loss, and its rounding noise interacts with differential privacy noise in ways we characterise empirically. [2]
- Communication cost grows linearly with the number of bins, and using 32 bins instead of 256 reduced traffic by 87 percent with an accuracy drop of only 0.3 percent. [3]
- Communication was reduced by a factor of 120 compared with histogram-based federated boosting. [4]

### SQ2: What accuracy, privacy and efficiency results are reported on how federated gradient boosting methods trade off privacy, accuracy and communication cost?

- Differentially private federated boosting is viable for predictive maintenance at moderate privacy budgets, but the accuracy cost at strong budgets and non-IID data needs further study. [1]
- Quantisation cuts communication by 75 percent with less than 0.5 percent accuracy loss, and its rounding noise interacts with differential privacy noise in ways we characterise empirically. [2]
- Communication cost grows linearly with the number of bins, and using 32 bins instead of 256 reduced traffic by 87 percent with an accuracy drop of only 0.3 percent. [3]
- Communication was reduced by a factor of 120 compared with histogram-based federated boosting. [4]

### SQ3: What limitations and open problems remain on how federated gradient boosting methods trade off privacy, accuracy and communication cost?

- Secure histogram aggregation makes horizontal federated boosting practical with near-centralised accuracy, but formal privacy guarantees for the released model remain an open question. [3]
- Differentially private federated boosting is viable for predictive maintenance at moderate privacy budgets, but the accuracy cost at strong budgets and non-IID data needs further study. [1]
- Leaf values and split thresholds allow an attacker to infer whether a record was used for training with up to 71 percent accuracy, and differential privacy with epsilon equal to 4 reduces attack accuracy to near random. [5]
- Quantisation cuts communication by 75 percent with less than 0.5 percent accuracy loss, and its rounding noise interacts with differential privacy noise in ways we characterise empirically. [2]

## Comparison across studies

- Several studies provide evidence on SQ1: Differentially private federated boosting is viable for predictive maintenance at moderate privacy budgets, but the… [1][2][3]
- Several studies provide evidence on SQ2: Differentially private federated boosting is viable for predictive maintenance at moderate privacy budgets, but the… [1][2][3]
- Several studies provide evidence on SQ3: Secure histogram aggregation makes horizontal federated boosting practical with near-centralised accuracy, but formal… [1][3][5]
- Reported results differ across settings: Quantisation cuts communication by 75 percent with less than 0.5 percent accuracy loss,… vs. Communication cost grows linearly with the number of bins, and using 32 bins instead of… [2][3]

## Research gaps and future directions

- **Differentially private federated boosting is viable for predictive maintenance at moderate privacy budgets, but the accuracy cost at…** Design experiments that address this limitation directly. [1]
- **Secure histogram aggregation makes horizontal federated boosting practical with near-centralised accuracy, but formal privacy guarantees…** Design experiments that address this limitation directly. [3]

## Limitations of this review

This report was generated in offline demo mode from bundled demo papers.

## Method comparison

| Study | Approach | Data / setting | Key result |
|---|---|---|---|
| [3] Rao et al. (2021) | Communication cost grows linearly with the number of bins, and using 32 bins instead of 256 reduced traffic… | see paper | Communication cost grows linearly with the number of bins, and using 32 bins instead of 256 reduced traffic… |
| [1] Silva et al. (2023) | Differentially private federated boosting is viable for predictive maintenance at moderate privacy budgets,… | see paper | Differentially private federated boosting is viable for predictive maintenance at moderate privacy budgets,… |
| [4] Petrova et al. (2022) | Communication was reduced by a factor of 120 compared with histogram-based federated boosting. | see paper | Communication was reduced by a factor of 120 compared with histogram-based federated boosting. |
| [5] Lee et al. (2023) | Leaf values and split thresholds allow an attacker to infer whether a record was used for training with up… | see paper | Leaf values and split thresholds allow an attacker to infer whether a record was used for training with up… |
| [2] Hassan et al. (2024) | Quantisation cuts communication by 75 percent with less than 0.5 percent accuracy loss, and its rounding… | see paper | Quantisation cuts communication by 75 percent with less than 0.5 percent accuracy loss, and its rounding… |

## References

[1] Marta Silva, Ravi Kumar. **Differentially Private Federated XGBoost for Predictive Maintenance**. *Demo Conference on Industrial AI (fictional)*. 2023.

[2] Noor Hassan, Felix Wagner. **Quantised Gradient Sharing for Federated Decision Trees on Edge Devices**. *Demo Conference on Edge Computing (fictional)*. 2024.

[3] Asha Rao, Kenji Mori, Lena Fischer. **Horizontal Federated Gradient Boosting with Secure Histogram Aggregation**. *Demo Workshop on Private ML (fictional)*. 2021.

[4] Elena Petrova, Samuel Okafor. **Communication-Efficient Federated Gradient Boosting via Local Tree Ensembles**. *Demo Workshop on Efficient ML (fictional)*. 2022.

[5] Hannah Lee, Diego Alvarez. **Membership Inference Attacks on Federated Tree Ensembles**. *Demo Conference on ML Security (fictional)*. 2023.

<details><summary>Evidence appendix: verified claims and the exact quotes they rest on</summary>

- **F1** [1] (SQ1, supported) Differentially private federated boosting is viable for predictive maintenance at moderate privacy budgets, but the accuracy cost at strong budgets and non-IID data needs further study.  
  > "Differentially private federated boosting is viable for predictive maintenance at moderate privacy budgets, but the accuracy cost at strong budgets and non-IID data needs further study." — `demo:0002#5`
- **F2** [2] (SQ1, supported) Quantisation cuts communication by 75 percent with less than 0.5 percent accuracy loss, and its rounding noise interacts with differential privacy noise in ways we characterise empirically.  
  > "Quantisation cuts communication by 75 percent with less than 0.5 percent accuracy loss, and its rounding noise interacts with differential privacy noise in ways we characterise empirically." — `demo:0008#0`
- **F3** [3] (SQ1, supported) Communication cost grows linearly with the number of bins, and using 32 bins instead of 256 reduced traffic by 87 percent with an accuracy drop of only 0.3 percent.  
  > "Communication cost grows linearly with the number of bins, and using 32 bins instead of 256 reduced traffic by 87 percent with an accuracy drop of only 0.3 percent." — `demo:0001#3`
- **F4** [4] (SQ1, supported) Communication was reduced by a factor of 120 compared with histogram-based federated boosting.  
  > "Communication was reduced by a factor of 120 compared with histogram-based federated boosting." — `demo:0004#2`
- **F5** [1] (SQ2, supported) Differentially private federated boosting is viable for predictive maintenance at moderate privacy budgets, but the accuracy cost at strong budgets and non-IID data needs further study.  
  > "Differentially private federated boosting is viable for predictive maintenance at moderate privacy budgets, but the accuracy cost at strong budgets and non-IID data needs further study." — `demo:0002#5`
- **F6** [2] (SQ2, supported) Quantisation cuts communication by 75 percent with less than 0.5 percent accuracy loss, and its rounding noise interacts with differential privacy noise in ways we characterise empirically.  
  > "Quantisation cuts communication by 75 percent with less than 0.5 percent accuracy loss, and its rounding noise interacts with differential privacy noise in ways we characterise empirically." — `demo:0008#0`
- **F7** [3] (SQ2, supported) Communication cost grows linearly with the number of bins, and using 32 bins instead of 256 reduced traffic by 87 percent with an accuracy drop of only 0.3 percent.  
  > "Communication cost grows linearly with the number of bins, and using 32 bins instead of 256 reduced traffic by 87 percent with an accuracy drop of only 0.3 percent." — `demo:0001#3`
- **F8** [4] (SQ2, supported) Communication was reduced by a factor of 120 compared with histogram-based federated boosting.  
  > "Communication was reduced by a factor of 120 compared with histogram-based federated boosting." — `demo:0004#2`
- **F9** [3] (SQ3, supported) Secure histogram aggregation makes horizontal federated boosting practical with near-centralised accuracy, but formal privacy guarantees for the released model remain an open question.  
  > "Secure histogram aggregation makes horizontal federated boosting practical with near-centralised accuracy, but formal privacy guarantees for the released model remain an open question." — `demo:0001#5`
- **F10** [1] (SQ3, supported) Differentially private federated boosting is viable for predictive maintenance at moderate privacy budgets, but the accuracy cost at strong budgets and non-IID data needs further study.  
  > "Differentially private federated boosting is viable for predictive maintenance at moderate privacy budgets, but the accuracy cost at strong budgets and non-IID data needs further study." — `demo:0002#5`
- **F11** [5] (SQ3, supported) Leaf values and split thresholds allow an attacker to infer whether a record was used for training with up to 71 percent accuracy, and differential privacy with epsilon equal to 4 reduces attack accuracy to near random.  
  > "Leaf values and split thresholds allow an attacker to infer whether a record was used for training with up to 71 percent accuracy, and differential privacy with epsilon equal to 4 reduces attack accuracy to near random." — `demo:0005#0`
- **F12** [2] (SQ3, supported) Quantisation cuts communication by 75 percent with less than 0.5 percent accuracy loss, and its rounding noise interacts with differential privacy noise in ways we characterise empirically.  
  > "Quantisation cuts communication by 75 percent with less than 0.5 percent accuracy loss, and its rounding noise interacts with differential privacy noise in ways we characterise empirically." — `demo:0008#0`

</details>
