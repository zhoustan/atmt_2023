import re
import matplotlib.pyplot as plt
import numpy as np

latex_data = ("""
\textbf{1} & 17.1 & 1.0 \\
\textbf{2} & 19.0 & 1.0 \\
\textbf{3} & 19.1 & 1.0 \\
\textbf{4} & 18.6 & 1.0 \\
\textbf{5} & 20.0 & 1.0 \\
\textbf{6} & 19.7 & 1.0 \\
\textbf{7} & 20.9 & 1.0 \\
\textbf{8} & 20.8 & 1.0 \\
\textbf{9} & 22.0 & 1.0 \\
\textbf{10} & 22.2 & 1.0 \\
\textbf{11} & 22.6 & 1.0 \\
\textbf{12} & 22.2 & 1.0 \\
\textbf{13} & 22.7 & 0.988 \\
\textbf{14} & 22.6 & 0.975 \\
\textbf{15} & 22.1 & 0.949 \\
\textbf{16} & 22.1 & 0.934 \\
\textbf{17} & 21.9 & 0.920 \\
\textbf{18} & 21.9 & 0.920 \\
\textbf{19} & 21.8 & 0.873 \\
\textbf{20} & 21.8 & 0.873 \\
\textbf{21} & 21.5 & 0.848 \\
\textbf{22} & 21.5 & 0.842 \\
\textbf{23} & 21.3 & 0.831 \\
\textbf{24} & 21.2 & 0.820 \\
\textbf{25} & 21.4 & 0.809 \\
""".replace('\textbf', ''))

data = []
pattern = re.compile(r"{(\d+)} & (\d+\.\d+) & (\d+\.\d+)")
matches = pattern.findall(latex_data)
for match in matches:
    beam_size = match[0]
    bleu = float(match[1])
    bp = float(match[2])
    data.append({"Beam size": beam_size, "BLEU": bleu, "BP": bp})

hyper_tuned_data_bleu = np.array([18., 19.2, 20., 20., 20.3, 20.1, 20.1, 20., 19.7, 19.7, 19.6,
                                  19.5, 19.6, 19.7, 19.8, 19.9, 19.8, 19.9, 19.9, 19.7, 19.7, 19.8,
                                  19.7, 19.8, 19.7])
hyper_tuned_data_bp = np.array(['1.000', '0.949', '0.920', '0.909', '0.897', '0.894', '0.883',
                                '0.874', '0.863', '0.861', '0.861', '0.856', '0.850', '0.849',
                                '0.844', '0.842', '0.833', '0.836', '0.834', '0.827', '0.824',
                                '0.821', '0.811', '0.821', '0.811'], dtype=np.float64)

fig, ax1 = plt.subplots()

ax1.set_xlabel('Beam size')
ax1.set_ylabel('BLEU Score', color='tab:blue')
ax1.plot([item["Beam size"] for item in data], [item["BLEU"] for item in data], color='tab:blue', marker='o',
         label='Baseline Model BLEU Score')
ax1.plot([item["Beam size"] for item in data], hyper_tuned_data_bleu, color='tab:blue', marker='s',
         label='Hypertuned Model BLEU Score')

ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Brevity Penalty', color='tab:red')
ax2.plot([item["Beam size"] for item in data], [item["BP"] for item in data], color='tab:red', marker='o',
         label='Baseline Model Brevity Penalty')
ax2.plot([item["Beam size"] for item in data], hyper_tuned_data_bp, color='tab:red', marker='s',
         label='Hypertuned Model Brevity Penalty')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.14, 0.3), fontsize='small')  # Adjust bbox_to_anchor to control legend position
plt.show()
