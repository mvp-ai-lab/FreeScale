import json
import os
import random

def extract_evaluation_large(scene_id, view_graph, num_views=6, evaluation_index={}):
    # evaluation_index = {}

    # only training node
    new_graph = {}
    for k, v in view_graph.items():
        if 'fv' in str(k):
            continue
        new_v = []
        for neigh in v:
            if 'fv' not in str(neigh[0]):
                new_v.append(neigh)
        new_graph[k] = new_v

    train_nodes = {nk: sum(item[1] for item in nv) for nk, nv in new_graph.items()}
    available_nodes = {
                    nk: weight_sum 
                    for nk, weight_sum in train_nodes.items() 
                    if len(new_graph.get(nk, [])) >= num_views - 1
                }
    nk_sort_list = sorted(
                    available_nodes.keys(), 
                    key=lambda nk: available_nodes[nk], 
                    reverse=True
                )
    
    nodes = random.sample(nk_sort_list, min(len(nk_sort_list), 10))
    success = 0
    for i, nk in enumerate(nodes):
        if success >= 5:
            break
        select_neigh = random.sample(new_graph[nk], num_views-1)
        image_index = [int(nk)] + [int(n[0]) for n in select_neigh]

        if max(image_index) - min(image_index) > 10:
            continue

        select = {
            "context":image_index[:num_context],
            "target": image_index[num_context:]
        }
        evaluation_index[success].update({scene_id: select})
        success += 1
    return evaluation_index

def extract_evaluation_small(scene_id, view_graph, num_views=6, existing_evaluation_index={}, max_records=4):
    # evaluation_index = {}

    # only training node
    new_graph = {}
    for k, v in view_graph.items():
        if 'fv' in str(k):
            continue
        new_v = []
        for neigh in v:
            if 'fv' not in str(neigh[0]):
                new_v.append(neigh)
        new_graph[k] = new_v

    train_nodes = {nk: sum(item[1] for item in nv) for nk, nv in new_graph.items()}
    available_nodes = {
                    nk: weight_sum 
                    for nk, weight_sum in train_nodes.items() 
                    if len(new_graph.get(nk, [])) >= num_views - 1
                }
    nk_sort_list = sorted(
                    available_nodes.keys(), 
                    key=lambda nk: available_nodes[nk], 
                    reverse=True
                )
    
    if scene_id not in existing_evaluation_index:
        existing_evaluation_index[scene_id] = []
    current_success_count = len(existing_evaluation_index[scene_id])

    nodes = random.sample(nk_sort_list, min(len(nk_sort_list), 100))
    for nk in nodes:
        if current_success_count >= max_records:
            return existing_evaluation_index
        for _ in range(5):
            if len(new_graph.get(nk, [])) < num_views - 1:
                break 
            select_neigh = random.sample(new_graph[nk], num_views - 1)
            image_index = [int(nk)] + [int(n[0]) for n in select_neigh]

            if max(image_index) - min(image_index) <= 20:
                select = {
                    "context": image_index[:num_context],
                    "target": image_index[num_context:]
                }
                
                existing_evaluation_index[scene_id].append(select)
                current_success_count += 1
            
                if current_success_count >= max_records:
                    return existing_evaluation_index
                
                break 
    if len(existing_evaluation_index[scene_id]) < max_records:
        fill_count = max_records - len(existing_evaluation_index[scene_id])
        if len(existing_evaluation_index[scene_id]) == 0:
            print(scene_id)
            existing_evaluation_index[scene_id] = [{'context': [129, 133, 134, 125], 'target': [124, 137]}]
        last_record = existing_evaluation_index[scene_id][-1]
        existing_evaluation_index[scene_id] = existing_evaluation_index[scene_id].extend([last_record] * fill_count)

    return existing_evaluation_index


    

# data_path = "exps/dl3dv_bench/out_active"
# num_context = 4
# num_target = 2
# num_views = num_context + num_target
# evaluation_index = {}
# max_records = 4
# # for i in range(5):
# #     evaluation_index[i] = {}

# for expname in os.listdir(data_path):
#     if not 'gs_nvs' in expname:
#         continue
#     scene_id = expname.split("_")[3]
#     # if scene_id in ["f477ffc4b398bed8e0d921f0fba9825ca63f317381c535c84be23be991ae1d7a"]:
#     #     continue
#     view_graph_path = f"{data_path}/{expname}/renders/view_graph.json"
#     if os.path.exists(view_graph_path):
#         view_graph = json.load(open(view_graph_path, "r")) 
#         evaluation_index = extract_evaluation_small(
#             scene_id, view_graph, existing_evaluation_index=evaluation_index, 
#             max_records=max_records
#         )

# # for k, index in evaluation_index.items():
# #     with open(f'freescale/lvsm/data/dl3dv/bench_small_{k}.json', 'w') as f:
# #         json.dump(index, f)

# for i in range(max_records):
#     one_group_per_scene = {
#         scene_id: results[i]
#         for scene_id, results in evaluation_index.items()
#         if results 
#     }

#     with open(f'freescale/lvsm/data/dl3dv/bench_small_{i}.json', 'w') as f:
#         json.dump(one_group_per_scene, f)


def sample_mipnerf(data_path, max_range=15, sequence_length=6, num_context=4):
    evaluation_index = {}
    for scene in os.listdir(data_path):
        N = len(os.listdir(os.path.join(data_path, scene, "images_4")))
        effective_range = min(max_range + 1, N + 1)
        max_min_index = N - effective_range + 1
        if max_min_index < 0:
            max_min_index = 0

        min_index = random.randint(0, max_min_index)
        upper_bound = min(min_index + max_range, N)
        available_indices = list(range(min_index, upper_bound + 1))
        sampled_sequence = random.sample(available_indices, sequence_length)

        evaluation_index[scene] = {
                    "context": sampled_sequence[:num_context],
                    "target": sampled_sequence[num_context:]
                }

    return evaluation_index


# json_dict = sample_mipnerf("/cephyr/users/qingwenz/Alvis/data/tanks_and_temples/")
json_dict = sample_mipnerf("/data/mipnerf360/")
with open(f'freescale/lvsm/data/mipnerf3603.json', 'w') as f:
    json.dump(json_dict, f)

