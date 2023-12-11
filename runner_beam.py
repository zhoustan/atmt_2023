import subprocess



def translate_with_beam_from_Fr_to_En(k):
    translate_command = ['python', 'translate_beam.py',
                        '--data', 'data/en-fr/prepared',
                        '--dicts', 'data/en-fr/prepared',
                        '--checkpoint-path', 'assignments/03/baseline/checkpoints/checkpoint_last.pt',
                        '--output', 'assignments/03/baseline/translations/FrToEn_translation.txt',
                        '--cuda', 'CUDA',
                        '--batch-size=512',
                        f'--beam-size={k}'
                        ]
    result = subprocess.run(translate_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f'Translation FrToEn with beam {k} : DONE')



def postprocess(k):
    postprocess_command = ['bash', 'scripts/postprocess.sh', 
                            'assignments/03/baseline/translations/FrToEn_translation.txt', 
                            'assignments/03/baseline/translations/FrToEn_translation.p.txt',
                            'en']

    result = subprocess.run(postprocess_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f'Postprocess FrToEn with beam {k} : DONE')

def delete_translation_files():
    # delete files producted by a run
    del_files_paths = [
        'assignments/03/baseline/translations/FrToEn_translation.txt',
        'assignments/03/baseline/translations/FrToEn_translation.p.txt'
        ]

    for file_path in del_files_paths:
        try:
            subprocess.run(['rm', file_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error deleting {file_path}: {e}")

def evaluate():
    # evaluate
    evaluate_command = ['cat assignments/03/baseline/translations/FrToEn_translation.p.txt | sacrebleu data/en-fr/raw/test.en']

    result = subprocess.run(evaluate_command, shell=True, capture_output=True, text=True)
    # print(result.stdout)

    with open('assignments/03/baseline/bleu_in_different_beam.txt', 'a') as f:
        f.write(result.stdout)
        f.write(result.stderr)
        print(result.stdout)



def main(k_ls):
    for k in k_ls:
        translate_with_beam_from_Fr_to_En(k)
        postprocess(k)
        evaluate()
        delete_translation_files()




if __name__ == '__main__':
    k_ls = list(range(21, 26))
    main(k_ls)