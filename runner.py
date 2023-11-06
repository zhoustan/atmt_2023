import subprocess


def main(batch_sizes, learning_rates):
    batch_sizes = batch_sizes
    learning_rates = learning_rates

    # 遍历所有参数组合
    for batch_size in batch_sizes:
        for lr in learning_rates:
            # train
            train(lr, batch_size)

            # translate FROM Fr To En
            print('Start Translating')
            translate_from_Fr_to_En()
            print('Translation FrToEn DONE')

            # postprocess 
            postprocess()

            # evaluate
            evaluate()

            # delete model and files produced by this model
            delete_files_produced_by_mtproj()
            
            # 提取结果记录到结果文件 
            # 记录格式：batch_size, lr, train_loss, valid_loss
            # with open('output_results.txt', 'a') as f:
            #     f.write(f"{batch_size}, {lr}, ")
            #     f.write(result.stdout.split('\n')[-3])
            #     f.write('\n')



def train(lr, batch_size):
    # 构造训练模型命令
    train_command = [
        'python', 'train.py',
        '--data', 'data/en-fr/prepared',
        '--source-lang', 'fr',
        '--target-lang', 'en',
        '--save-dir', 'assignments/03/baseline/checkpoints',
        f'--lr={lr}',
        f'--batch-size={batch_size}',
        # '--train-on-tiny',
        '--cuda']

    # 运行脚本并捕获输出
    result = subprocess.run(train_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # 打印和记录输出
    print(f"Running with --batch-size={batch_size} and --lr={lr}")
    
    # 捕获输出并按行分割
    output_lines = result.stderr.splitlines()

    # 记录到文件
    with open('assignments/03/baseline/output_log_{}_{}.txt'.format(batch_size, str(lr).replace('.', '_')), 'a') as f:
        f.write(f"Running with --batch-size={batch_size} and --lr={lr}\n")
        f.write(result.stdout)
        f.write(result.stderr)



    # 获取最后三行
    last_three_lines = output_lines[-3:]

    with open(f'assignments/03/baseline/output_hyper_params_and_loss.txt', 'a') as f:
        f.write(f"\n{batch_size},{lr}\n")
        f.write('\n'.join(last_three_lines))
        f.write('\n')



def translate_from_Fr_to_En():
    translate_command = ['python', 'translate.py',
                        '--data', 'data/en-fr/prepared',
                        '--dicts', 'data/en-fr/prepared',
                        '--checkpoint-path', 'assignments/03/baseline/checkpoints/checkpoint_last.pt',
                        '--output', 'assignments/03/baseline/translations/FrToEn_translation.txt',
                        '--cuda',
                        '--batch-size=512'
                        ]

    result = subprocess.run(translate_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # with open('assignments/03/baseline/output_log_{}_{}.txt'.format(batch_size, str(lr).replace('.', '_')), 'a') as f:
    #     f.write(result.stdout)
    #     f.write(result.stderr)

    print('Translation FrToEn DONE')


def postprocess():
    postprocess_command = ['bash', 'scripts/postprocess.sh', 
                            'assignments/03/baseline/translations/FrToEn_translation.txt', 
                            'assignments/03/baseline/translations/FrToEn_translation.p.txt',
                            'en']

    result = subprocess.run(postprocess_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print('Postprocess FrToEn DONE')

def evaluate():
    # evaluate
    evaluate_command = ['cat assignments/03/baseline/translations/FrToEn_translation.p.txt | sacrebleu data/en-fr/raw/test.en']

    result = subprocess.run(evaluate_command, shell=True, capture_output=True, text=True)
    print(result.stdout)

    with open('assignments/03/baseline/blue_scores.txt', 'a') as f:
        f.write(result.stdout)
        f.write(result.stderr)


def delete_files_produced_by_mtproj():
    # delete files producted by a run
    del_files_paths = [
        'assignments/03/baseline/checkpoints/checkpoint_last.pt',
        'assignments/03/baseline/checkpoints/checkpoint_best.pt',
        'assignments/03/baseline/translations/FrToEn_translation.txt',
        'assignments/03/baseline/translations/FrToEn_translation.p.txt'
        ]

    for file_path in del_files_paths:
        try:
            subprocess.run(['rm', file_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error deleting {file_path}: {e}")


if __name__ == '__main__':

    # 参数的不同组合
    batch_sizes = [8, 32, 64, 128, 256, 512]
    learning_rates = [0.01, 0.001, 0.0006, 0.0004, 0.0002, 0.0001]

    # batch_sizes = [512, 256]
    # learning_rates = [0.001]
    main(batch_sizes, learning_rates)