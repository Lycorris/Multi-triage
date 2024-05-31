import os
import pandas as pd
import tqdm
corpus_vision = "05-31"
file_folder = f"./Data_{corpus_vision}"
target_folder = f"./Data_{corpus_vision}"

# 将文件夹下的所有文件合并成一个文件
def concat_files(file_folder):
    os.makedirs(f"./ConcatedData_{corpus_vision}", exist_ok=True)
    result = pd.DataFrame()
    for file in tqdm.tqdm(os.listdir(file_folder)):
        try:
            data = pd.read_csv(f"{file_folder}/{file}")
            result = pd.concat([result, data], ignore_index=True)
        except Exception as e:
            print(f"Error processing {file}...")
            print(e)
    print(len(result))
    result.to_csv(f"./ConcatedData_{corpus_vision}/concated_misc.csv", index=False, encoding='utf-8')

if __name__ == "__main__":
    concat_files(file_folder)
    # result = split_data(file_folder)
    # generate_report(result)