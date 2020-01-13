import json
import numpy as np
import matplotlib.pyplot as plt

json_folder = "../jsons/"
json_folder_2 = "../jsons2/"
figure_folder = "/home/broecker/src/udacity_rl/report_p1_2/figures/"

data_double_dqn = {
    "file_name": "run-DQN_buffer_NORMAL_lr_0.0003_skip_0_alpha_0.60_beta_start_0.40_end_1.00_steps_10000_tau_0.001_batch_size_64_gamma_0.95-tag-Avg_Reward.json",
    "label": "DDQN"}

data_dqn = {
    "file_name": "run-DQN_buffer_NORMAL_lr_0.0003_skip_0_alpha_0.60_beta_start_0.40_end_1.00_steps_10000_tau_1.0_batch_size_64_gamma_0.95-tag-Avg_Reward.json",
    "label": "DQN"}

data_double_duel = {
    "file_name": "run-DUEL_DQN_buffer_NORMAL_lr_0.0003_skip_0_alpha_0.60_beta_start_0.40_end_1.00_steps_10000_tau_0.001_batch_size_64_gamma_0.95-tag-Avg_Reward.json",
    "label": "Double Dual DQN"}

data_double_dqn_2 = {
    "file_name": "DOUBLE_DQN.json",
    "label": "DDQN"}

data_dqn_2 = {
    "file_name": "DQN.json",
    "label": "DQN"}

data_duel_2 = {
    "file_name": "DUEL_DQN.json",
    "label": "Dueling DQN"}

data_double_duel_2 = {
    "file_name": "DUELING_DOUBLE_DQN.json",
    "label": "Dueling DDQN"}


data_duel_dqn_priority = {
    "file_name": "run-DUEL_DQN_buffer_PRIORITY_lr_0.0003_skip_0_alpha_0.10_beta_start_0.00_end_0.40_steps_10000_tau_0.001_batch_size_64_gamma_0.95-tag-Avg_Reward.json",
    "label": "Duel DQN Priority Buffer"}

data_double_dqn_skip_1 = {
    "file_name": "run-DQN_buffer_NORMAL_lr_0.0003_skip_1_alpha_0.60_beta_start_0.40_end_1.00_steps_10000_tau_0.001_batch_size_64_gamma_0.95-tag-Avg_Reward.json",
    "label": "DDQN skip_frame(1)"}
data_double_dqn_skip_2 = {
    "file_name": "run-DQN_buffer_NORMAL_lr_0.0003_skip_2_alpha_0.60_beta_start_0.40_end_1.00_steps_10000_tau_0.001_batch_size_64_gamma_0.95-tag-Avg_Reward.json",
    "label": "DDQN skip_frame(2)"}

data_duel_dqn_priority_1 = {
    "file_name": "run-DUEL_DQN_buffer_PRIORITY_lr_0.0003_skip_0_alpha_0.10_beta_start_0.00_end_0.40_steps_10000_tau_0.001_batch_size_64_gamma_0.95-tag-Avg_Reward.json",
    "label": "Priority Buffer Beta(0.4), Alpha(0.1)"}

data_duel_dqn_priority_2 = {
    "file_name": "run-DUEL_DQN_buffer_PRIORITY_lr_0.0003_skip_0_alpha_0.40_beta_start_0.40_end_1.00_steps_10000_tau_0.001_batch_size_64_gamma_0.95-tag-Avg_Reward.json",
    "label": "Priority Buffer Beta(0.4), Alpha(0.4)"}
data_duel_dqn_priority_3 = {
    "file_name": "run-DUEL_DQN_buffer_PRIORITY_lr_0.0003_skip_0_alpha_0.60_beta_start_0.40_end_1.00_steps_10000_tau_0.001_batch_size_64_gamma_0.95-tag-Avg_Reward.json",
    "label": "Priority Buffer Beta(0.4), Alpha(0.6)"}

def plot_reward_per_step(data_set, folders, colors, save_name):
    labels = []
    for data_frame, color, folder in zip(data_set, colors, folders):
        labels.append(data_frame["label"])
        f = open(folder + data_frame["file_name"], 'r')

        data = np.array(json.load(f))
        xs = data[:, 1]
        ys = data[:, 2]
        firsts = [x for x, y in zip(xs, ys) if y > 13.]
        first = 'n/a' if not firsts else firsts[0]
        print("{} & {} &{:.2f} \\\\".format(data_frame["label"], first, max(ys)))
        print("\\hline")
        plt.plot(xs, ys, color=color, label=data_frame["label"], linewidth=1)
        f.close()
    plt.grid(color='#7f7f7f', linestyle='-', linewidth=1)
    plt.hlines(13, 0, 1600, linestyles='dashed', linewidth=2.5)
    plt.xlim([0., 1600])
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    plt.axes().set_aspect(aspect=40)
    plt.savefig(figure_folder + save_name, format="pdf", pad_inches=0, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    plot_reward_per_step([data_dqn_2, data_double_dqn_2, data_duel_2, data_double_duel_2], [json_folder_2] * 4 , ['r', 'b', 'g', 'k'], 'network_types.pdf')
    plot_reward_per_step([data_double_dqn_2, data_double_dqn_skip_1, data_double_dqn_skip_2], [json_folder_2]  + [json_folder] * 2, ['r', 'b', 'g'], 'skip_frames.pdf')
    plot_reward_per_step([data_duel_dqn_priority_1, data_duel_dqn_priority_2, data_duel_dqn_priority_3], [json_folder] * 3, ['r', 'b', 'g'], 'buffer.pdf')
    # plot_reward_per_step([data_double_dqn, data_double_dqn_skip], ['r', 'b'], '../figures/priory.pdf')
