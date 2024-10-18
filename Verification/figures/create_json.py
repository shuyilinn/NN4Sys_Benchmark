import copy
import os
import json

# List of models and verifiers to process
Models = ['pensieve', 'lindex', 'aurora', 'decima', 'bloom_filter', 'cardinality']
Verifiers = ['abcrown', 'marabou']

# Directory containing the ONNX models
ONND_DIR = "../../Benchmarks/onnx/"

# Mapping from specification names to model filenames
spec_to_model_map = {"aurora_big_101": "aurora_big_simple.onnx",
                     "aurora_big_102": "aurora_big_simple.onnx",
                     "aurora_big_2": "aurora_big_simple.onnx",
                     "aurora_big_3": "aurora_big_parallel.onnx",
                     "aurora_big_4": "aurora_big_concat.onnx",

                     "aurora_small_101": "aurora_small_simple.onnx",
                     "aurora_small_102": "aurora_small_simple.onnx",
                     "aurora_small_2": "aurora_small_simple.onnx",
                     "aurora_small_3": "aurora_small_parallel.onnx",
                     "aurora_small_4": "aurora_small_concat.onnx",

                     "aurora_mid_101": "aurora_mid_simple.onnx",
                     "aurora_mid_102": "aurora_mid_simple.onnx",
                     "aurora_mid_2": "aurora_mid_simple.onnx",
                     "aurora_mid_3": "aurora_mid_parallel.onnx",
                     "aurora_mid_4": "aurora_mid_concat.onnx",

                     "decima_mid_1": "decima_mid_simple.onnx",
                     "decima_mid_2": "decima_mid_simple.onnx",

                     "pensieve_big_1": "pensieve_big_simple.onnx",
                     "pensieve_big_2": "pensieve_big_simple.onnx",
                     "pensieve_big_3": "pensieve_big_parallel.onnx",
                     "pensieve_small_1": "pensieve_small_simple.onnx",
                     "pensieve_small_2": "pensieve_small_simple.onnx",
                     "pensieve_small_3": "pensieve_small_parallel.onnx",
                     "pensieve_mid_1": "pensieve_mid_simple.onnx",
                     "pensieve_mid_2": "pensieve_mid_simple.onnx",
                     "pensieve_mid_3": "pensieve_mid_parallel.onnx",

                     "mscn_128d": "mscn_128d.onnx",
                     "mscn_128d_dual": "mscn_128d_dual.onnx",
                     "mscn_2048d_dual": "mscn_2048d_dual.onnx",
                     "mscn_2048d": "mscn_2048d.onnx",

                     "lindex_0": "lindex.onnx",
                     "lindex_1": "lindex.onnx",
                     "lindex_2": "lindex.onnx",

                     "linde_deep_0": "lindex_deep.onnx",
                     "linde_deep_1": "lindex_deep.onnx",
                     "linde_deep_2": "lindex_deep.onnx"

                     }


def calculate_avg_time(dic1, dic2, dic3, times1, times2, time3):
    """
    Calculate the average time for each key based on the provided dictionaries.
    """
    ret = {}
    for key in times2:
        if key in times1:
            times1[key] += times2[key]
        else:
            times1[key] = times2[key]
    for key in time3:
        if key in times1:
            times1[key] += time3[key]
        else:
            times1[key] = time3[key]
    for key in dic2:
        if key in dic1:
            dic1[key] += dic2[key]
        else:
            dic1[key] = dic2[key]
    for key in dic3:
        if key in dic1:
            dic1[key] += dic3[key]
        else:
            dic1[key] = dic3[key]
    for key in dic1:
        if dic1[key] == 10:
            ret[key] = times1[key] / dic1[key]
        else:
            timeout = 10 - dic1[key]
            total_time = timeout * 180 + times1[key]
            ret[key] = total_time / 10

    return ret


def init_dic():
    """
    Initialize a dictionary with default values.
    """
    ret = {
        'safe': 0,
        'unsafe': 0,
        'time': 0,
        'timeout': 0
    }
    return ret


def main():
    """
    Main function to process data and generate evaluation results.
    """
    datas = {}
    for verifier in Verifiers:
        for model in Models:
            dir = f'../{model}_{verifier}_running_result'
            if not os.path.exists(dir):
                continue
            files = os.listdir(dir)

            # Initialize counters and dictionaries
            unsat = 0
            sat = 0
            sat_dic = {}
            unsat_dic = {}
            sat_time = {}
            unsat_time = {}
            timeout_dic = {}
            timeout_time = {}

            # Process each file in the directory
            for f in files:
                file = f'{dir}/' + f
                if file[-3:] != 'txt':
                    continue
                index = '_'.join(f[:-4].split('_')[:-1])
                if 'pensieve' in index or 'aurora' in index:
                    index = '_'.join(f.split('_')[:-3])

                timeout = -1

                # Read and process each result file
                with open(file, 'r') as f:
                    result = 'timeout'
                    if verifier == 'abcrown':
                        for line in f:
                            line = line.strip()
                            if line.startswith("Result"):
                                result = line[8:]
                            if line.startswith("Time"):
                                timeout = float(line[6:15])
                    else:
                        for line in f:
                            line = line.strip()
                            if line.startswith("sat"):
                                result = "sat"
                            if line.startswith("unsat"):
                                result = "unsat"
                            if line.startswith("Time"):
                                timeout = float(line[5:15])

                    if timeout > 180:
                        result = "timeout"
                        timeout = 180

                    timeout = float(timeout)
                    # Update counts and times based on the result
                    if result == 'unsat':
                        unsat += 1
                        unsat_dic[index] = unsat_dic.get(index, 0) + 1
                        unsat_time[index] = unsat_time.get(index, 0) + timeout
                    elif result == 'sat':
                        sat += 1
                        sat_dic[index] = sat_dic.get(index, 0) + 1
                        sat_time[index] = sat_time.get(index, 0) + timeout
                    else:
                        timeout_dic[index] = timeout_dic.get(index, 0) + 1
                        timeout_time[index] = timeout_time.get(index, 0) + 180

            # Prepare dictionaries for average time calculation
            sat_dic = dict(sorted(sat_dic.items()))
            sat_dic_copy = copy.deepcopy(sat_dic)
            unsat_dic = dict(sorted(unsat_dic.items()))
            unsat_dic_copy = copy.deepcopy(unsat_dic)
            timeout_dic_copy = copy.deepcopy(timeout_dic)

            # Calculate average times
            avg_time = calculate_avg_time(
                sat_dic_copy, unsat_dic_copy, timeout_dic_copy, sat_time, unsat_time, timeout_time
            )

            # Compile data into the main datas dictionary
            for key in avg_time:
                if key not in datas:
                    datas[key] = {}
                if verifier not in datas[key]:
                    datas[key][verifier] = init_dic()

                datas[key][verifier]['time'] = avg_time[key]

                datas[key][verifier]['unsafe'] = sat_dic.get(key, 0)
                datas[key][verifier]['safe'] = unsat_dic.get(key, 0)
                datas[key][verifier]['timeout'] = timeout_dic.get(key, 0)

                # Ensure total runs count to 10
                total_runs = (datas[key][verifier]['unsafe'] +
                              datas[key][verifier]['safe'] +
                              datas[key][verifier]['timeout'])
                if total_runs < 10:
                    datas[key][verifier]['timeout'] += 10 - total_runs

            # Add model size information
            for key in datas:
                try:
                    size = os.path.getsize(ONND_DIR + spec_to_model_map[key])
                    datas[key]["size"] = size
                except:
                    continue

    # Optional: Print the data for verification
    # for key in datas.keys():
    #     print(key)
    #     print(datas[key])

    # Sort the datas dictionary
    datas = dict(sorted(datas.items()))

    # Convert datas to JSON format and write to file
    datas_json = json.dumps(datas)
    with open('eval_results.json', 'w') as f:
        f.write(datas_json)

    # Print that the creation is finished
    print('Creation is finished.')



if __name__ == "__main__":
    main()
