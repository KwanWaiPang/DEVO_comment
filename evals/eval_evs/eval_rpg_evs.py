import os
import torch
from devo.config import cfg

from utils.load_utils import load_gt_us, rpg_evs_iterator
from utils.eval_utils import assert_eval_config, run_voxel
from utils.eval_utils import log_results, write_raw_results, compute_median_results
from utils.viz_utils import viz_flow_inference

@torch.no_grad()
def evaluate(config, args, net, train_step=None, datapath="", split_file=None,
             stride=1, trials=1, plot=False, save=False, return_figure=False, viz=False, timing=False, side='left', viz_flow=False):
    dataset_name = "rpg_evs"
    assert side == "left" or side == "right"

    if config is None:
        config = cfg
        config.merge_from_file("config/eval_rpg.yaml")

    scenes = open(split_file).read().split()
    scenes = [s for s in scenes if '#' not in s]
    print("the number of scenes is", len(scenes),"the input scenes are: ", scenes)

    results_dict_scene, figures = {}, {}
    all_results = []
    for i, scene in enumerate(scenes):
        if "simulation_3planes" in scene:
            H, W = 260, 346
            pose_freq = 1000
        else:
            H, W = 180, 240
            pose_freq = 200

        traj_hf_path = os.path.join(datapath, scene, f"gt_stamped_{side}.txt")
        if not os.path.exists(traj_hf_path):
            print(f"scene {scene} has no GT, skipping")
            continue

        print(f"Eval on {scene}")
        results_dict_scene[scene] = []

        for trial in range(trials):
            datapath_val = os.path.join(datapath, scene)

            # run the slam system
            traj_est, tstamps, flowdata = run_voxel(datapath_val, config, net, viz=viz, 
                                          iterator=rpg_evs_iterator(datapath_val, side=side, stride=stride, timing=timing, dT_ms=None, H=H, W=W), # optionally pass DELTA_MS
                                          timing=timing, H=H, W=W, viz_flow=viz_flow)

            # load traj
            tss_traj_us, traj_hf = load_gt_us(traj_hf_path)
 
            # do evaluation 
            data = (traj_hf, tss_traj_us, traj_est, tstamps)
            hyperparam = (train_step, net, dataset_name, scene, trial, cfg, args)
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                   plot=plot, save=save, return_figure=return_figure, stride=stride,
                                                                   expname=args.expname)
            
            if viz_flow:
                viz_flow_inference(outfolder, flowdata)
            
        print(scene, sorted(results_dict_scene[scene]))

    # write output to file with timestamp
    write_raw_results(all_results, outfolder)
    results_dict = compute_median_results(results_dict_scene, all_results, dataset_name, outfolder)
        
    if return_figure:
        return results_dict, figures
    return results_dict, None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/eval_rpg.yaml")
    parser.add_argument('--datapath', default='', help='path to dataset directory')
    parser.add_argument('--weights', default="DEVO.pth")
    parser.add_argument('--val_split', type=str, default="splits/rpg/rpg_val.txt")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--return_figs', action="store_true")
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--timing', action="store_true")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--side', type=str, default="left")
    parser.add_argument('--viz_flow', action="store_true")
    parser.add_argument('--expname', type=str, default="")

    args = parser.parse_args()
    assert_eval_config(args)

    cfg.merge_from_file(args.config)
    print("Running eval_event_camera_dataset.py with config...")
    print(cfg)

    torch.manual_seed(1234)
    
    args.plot = True
    args.save_trajectory = True
    # args.viz_flow = True
    val_results, val_figures = evaluate(cfg, args, args.weights, datapath=args.datapath, split_file=args.val_split, trials=args.trials, \
                       plot=args.plot, save=args.save_trajectory, return_figure=args.return_figs, viz=args.viz, timing=args.timing, \
                        side=args.side, stride=args.stride, viz_flow=args.viz_flow)
    
    print("val_results= \n")
    for k in val_results:
        print(k, val_results[k])
