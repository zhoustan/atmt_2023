from subprocess import Popen, PIPE
import os

file_template = "fr_en_translations_beamsize_13_alpha_02_gamma_%s"
# file_template = "fr_en_translations_beamsize_13_gamma_%s"


run_beam_template = """python translate_adapted_beam.py --data data/en-fr/prepared --dicts data/en-fr/prepared --checkpoint-path assignments/05/baseline/checkpoints/checkpoint_best.pt --output assignments/05/baseline/%s.txt --cuda True --batch-size 512 --beam-size 13 --n-best 3 --alpha 0.2 --gamma %s"""


# def run_cygwin(cmd):
#     p = Popen(r"C:/cygwin64/Cygwin.bat", stdin=PIPE, stdout=PIPE)
#     cmd = "cd /cygdrive/g/My\ Drive/uzh/23HS/ATMT/assignments/Assignment03 && " + cmd
#     print(cmd)
#     p.stdin.write(cmd.encode('utf-8'))
#     p.stdin.close()
#     out = p.stdout.read()
#     print(out.decode())


cmd_template = """bash scripts/postprocess.sh assignments/05/baseline/fr_en_translations_beamsize_13_alpha_02_gamma_%s_n%s.txt assignments/05/baseline/fr_en_translations_beamsize_13_alpha_02_gamma_%s_n%s.p.txt en"""
# cmd_template = """bash scripts/postprocess.sh assignments/05/baseline/fr_en_translations_beamsize_13_gamma_%s_n%s.txt assignments/05/baseline/fr_en_translations_beamsize_13_gamma_%s_n%s.p.txt en"""

eval_template = """type assignments/05/baseline/fr_en_translations_beamsize_13_alpha_02_gamma_%s_n%s.p.txt  | sacrebleu data/en-fr/raw/test.en """
# eval_template = """type assignments/05/baseline/fr_en_translations_beamsize_13_gamma_%s_n%s.p.txt  | sacrebleu data/en-fr/raw/test.en """

for gamma in ["16", "17", "18", "19"]:
    # print("==============gamma:%s==============" % gamma)
    file = file_template % gamma
    # print(file)
    run_beam = run_beam_template % (file, "0."+gamma)
    # os.system(run_beam)
    for i in range(3):
        cmd = cmd_template % (gamma, str(i + 1), gamma, str(i + 1))
        # print(cmd)
        eval_cmd = eval_template % (gamma, str(i + 1))
        print(eval_cmd)
        # os.system(eval_cmd)
