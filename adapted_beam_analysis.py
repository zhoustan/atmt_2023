from sacrebleu.metrics import BLEU

gamma = 14

file1 = open("assignments/05/baseline/fr_en_translations_beamsize_13_alpha_02_gamma_%s_n1.p.txt"%gamma, "r")
file2 = open("assignments/05/baseline/fr_en_translations_beamsize_13_alpha_02_gamma_%s_n2.p.txt"%gamma, "r")
file3 = open("assignments/05/baseline/fr_en_translations_beamsize_13_alpha_02_gamma_%s_n3.p.txt"%gamma, "r")

gt = open("data/en-fr/raw/test.en", "r")


one_best = file1.readlines()
two_best = file2.readlines()
three_best = file3.readlines()
gt = gt.readlines()

bleu = BLEU()
print(len(one_best))
for i in range(len(one_best)):
    # refs = [one_best[i], two_best[i], three_best[i]]
    # sys = [gt[i]]
    score1 = bleu.corpus_score([gt[i]], [one_best[i]])
    score2 = bleu.corpus_score([gt[i]], [two_best[i]])
    score3 = bleu.corpus_score([gt[i]], [three_best[i]])
    # print(score1)
    # print(score2)
    # print(score3)
    if float(str(score1).split(' ')[2]) > 1:

        # print(score1)
        print(one_best[i].strip())
        print(two_best[i].strip())
        print(three_best[i].strip())
        # print(gt[i].strip())
        print("="*50)