'''
This script is to draw diagrams for verification result visualization.

'''


import matplotlib.pyplot as plt
import json
import numpy as np


def adjust_map(map):
    """
    Adjusts the values in the map so that all strings are centered based on the longest string.
    """
    # Find the longest string in the map values
    max_length = max(len(value) for value in map.values())

    # Adjust all values to be centered based on the max_length
    for key, value in map.items():
        length = len(value)
        delta = max_length - length
        left = delta // 2
        right = delta - left
        map[key] = " " * left + value + " " * right

    return map


def main():
    # Dictionary that maps model names to their human-readable versions
    map = {
        'aurora_big_101': "CongestCtrl_large_spec101",
        'aurora_big_102': "CongestCtrl_large_spec102",
        'aurora_big_2': "CongestCtrl_large_spec2",
        'aurora_big_3': "CongestCtrl_large_dual_spec3",
        'aurora_big_4': "CongestCtrl_large_chain_spec4",
        'aurora_mid_101': "CongestCtrl_mid_spec101",
        'aurora_mid_102': "CongestCtrl_mid_spec102",
        'aurora_mid_2': "CongestCtrl_mid_spec2",
        'aurora_mid_3': "CongestCtrl_mid_dual_spec3",
        'aurora_mid_4': "CongestCtrl_mid_chain_spec4",
        'aurora_small_101': "CongestCtrl_small_spec101",
        'aurora_small_102': "CongestCtrl_small_spec102",
        'aurora_small_2': "CongestCtrl_small_spec2",
        'aurora_small_3': "CongestCtrl_small_dual_spec3",
        'aurora_small_4': "CongestCtrl_small_chain_spec4",
        'bloom_filter': "BloomFilter",
        'decima_mid_1': "LearnedSched_spec1",
        'decima_mid_2': "LearnedSched_spec2",
        'lindex_0': "LearnedIndex_spec1",
        'lindex_1': "LearnedIndex_spec2",
        'lindex_2': "LearnedIndex_spec3",
        'lindex_deep_0': "LearnedIndex_large_spec1",
        'lindex_deep_1': "LearnedIndex_large_spec2",
        'lindex_deep_2': "LearnedIndex_large_spec3",
        'mscn_128d': "CardEsti_small",
        'mscn_128d_dual': "CardEsti_small_dual",
        'mscn_2048d': "CardEsti_large",
        'mscn_2048d_dual': "CardEsti_large_dual",
        'pensieve_big_1': "AdaptBitrate_large_spec1",
        'pensieve_big_2': "AdaptBitrate_large_spec2",
        'pensieve_big_3': "AdaptBitrate_large_dual_spec3",
        'pensieve_mid_1': "AdaptBitrate_mid_spec1",
        'pensieve_mid_2': "AdaptBitrate_mid_spec2",
        'pensieve_mid_3': "AdaptBitrate_mid_dual_spec3",
        'pensieve_small_1': "AdaptBitrate_small_spec1",
        'pensieve_small_2': "AdaptBitrate_small_spec2",
        'pensieve_small_3': "AdaptBitrate_small_dual_spec3",
    }

    # Font settings for the plots
    font = {'family': 'DejaVu Sans', 'color': 'black', 'weight': 'normal', 'size': 15}
    font_text = {'family': 'DejaVu Sans', 'color': 'black', 'weight': 'normal', 'size': 14}
    font_legend = {'family': 'DejaVu Sans', 'size': 15}

    # Load the JSON results
    with open('eval_results.json') as f:
        datas = json.load(f)

    # Create a list of all keys (species)
    species = list(datas.keys())

    # Prepare data for the runtime plots
    marabou_can_dic_for_sort = {}
    marabou_cannot_dic_for_sort = {}

    # Separate datasets based on whether 'marabou' data is available
    for key in datas:
        if 'marabou' in datas[key]:
            marabou_can_dic_for_sort[key] = datas[key]['abcrown']['time']
        else:
            marabou_cannot_dic_for_sort[key] = datas[key]['abcrown']['time']

    # Sort the datasets by runtime
    marabou_can_dic_for_sort = dict(sorted(marabou_can_dic_for_sort.items(), key=lambda x: x[1]))
    marabou_cannot_dic_for_sort = dict(sorted(marabou_cannot_dic_for_sort.items(), key=lambda x: x[1]))

    # Prepare runtime and annotations for the bar plot
    time_sets = {'abcrown': [], 'marabou': []}
    annotates = {'abcrown': [], 'marabou': []}
    x_ticks = []

    # Fill the lists with sorted data
    for key in marabou_can_dic_for_sort:
        time_sets['abcrown'].append(round(datas[key]['abcrown']['time'], 2))
        time_sets['marabou'].append(round(datas[key]['marabou']['time'], 2))
        annotates['abcrown'].append(round(datas[key]['abcrown']['time'], 2))
        annotates['marabou'].append(round(datas[key]['marabou']['time'], 2))
        x_ticks.append(key)

    for key in marabou_cannot_dic_for_sort:
        time_sets['abcrown'].append(round(datas[key]['abcrown']['time'], 2))
        time_sets['marabou'].append(0)  # No marabou data
        annotates['abcrown'].append(round(datas[key]['abcrown']['time'], 2))
        annotates['marabou'].append('NaN')
        x_ticks.append(key)

    # Plot the runtime comparison for abcrown and marabou
    x = np.arange(len(species))  # Label locations
    width = 0.40  # Bar width
    fig, ax = plt.subplots(layout='constrained', figsize=(16, 4.5))

    # Bar plot for abcrown runtimes
    plt.bar(x + width / 2, time_sets['abcrown'], width, label=chr(945) + chr(946) + '-CROWN', edgecolor="black", linewidth=2)
    
    # Adjust positions to avoid overlap
    [plt.text(i + width / 2, time_sets['abcrown'][i], str(annotates['abcrown'][i]), ha='center', va='bottom', fontdict=font_text)
     for i in range(len(x_ticks))]

    # Bar plot for marabou runtimes
    ax.bar(x + width / 2 + width, time_sets['marabou'], width, label='Marabou', edgecolor="black", linewidth=2)
    [plt.text(i + width / 2 + width, max(time_sets['marabou'][i], 0.013), str(annotates['marabou'][i]), ha='center', va='bottom', fontdict=font_text)
     for i in range(len(x_ticks))]

    # Configure plot
    plt.xlim(left=-0.5, right=37)
    ax.legend(loc='upper left', ncols=3, prop=font_legend)
    ax.set_xticks(x + width, [map[i] for i in x_ticks], fontdict=font)
    ax.set_ylabel('Average Runtime (s)', fontdict=font)
    plt.tick_params(labelsize=15, labelfontfamily='DejaVu Sans')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    [spine.set_linewidth(2) for spine in ax.spines.values()]
    plt.xticks(rotation=-30, ha="left", rotation_mode="anchor")
    plt.yscale("log")
    plt.savefig("verification_runtime.pdf", format="pdf", bbox_inches="tight")
    print("[Done] Successfully generated verification_runtime.pdf")

    # Plot runtime vs model size
    fig, ax = plt.subplots(layout='constrained', figsize=(16, 4.5))
    size_dic = {}
    run_time_dic = {}

    # Collect runtime and size data for specific models
    for key in datas:
        if "pensieve" in key or "aurora" in key:
            if 'size' in datas[key]:
                size_dic[key] = datas[key]['size']
                run_time_dic[key] = datas[key]["abcrown"]["time"]

    # Sort models by size
    size_dic = dict(sorted(size_dic.items(), key=lambda x: x[1]))
    runtime = [run_time_dic[key] for key in size_dic]

    # Bar plot for runtimes
    names = [map[i] for i in size_dic.keys()]
    ax.bar(names, runtime, width=0.7, label='Verification Runtime', lw=2, ec='black')
    plt.xticks(rotation=-30, ha="left", rotation_mode="anchor", fontproperties='DejaVu Sans', size=15)

    # Plot model sizes as a line plot on a secondary axis
    ax2 = ax.twinx()
    ax2.plot(names, size_dic.values(), '-o', label='Model Size', color='orange')
    plt.xlim(left=-1, right=24)

    # Configure axes and labels
    ax2.set_ylabel('Size (byte)', fontdict=font)
    ax.set_ylabel('Average Runtime (s)', fontdict=font)
    plt.yscale("log")
    plt.tick_params(labelsize=15, labelfontfamily='DejaVu Sans')
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    fig.legend(loc=(0.04, 0.9), ncols=3, prop=font_legend)
    [spine.set_linewidth(2) for spine in ax.spines.values()]
    plt.savefig("verification_runtime_and_model_size.pdf", format="pdf", bbox_inches="tight")
    print("[Done] Successfully generated verification_runtime_and_model_size.pdf")


if __name__ == "__main__":
    main()
