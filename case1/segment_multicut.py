import subprocess


def segment_mc(block_id):
    ilastik_exe = "/home/pape/Work/software/src/ilastik/ilastik-1.4.0b21-Linux/run_ilastik.sh"
    ilp = "./segmentation.ilp"

    raw_path = f"./data/blocks/raw_block{block_id}.h5"
    pmap_path = f"./data/blocks/pred_block{block_id}.h5"
    out_path = f"./data/blocks/segmentation_block{block_id}.h5"
    cmd = [ilastik_exe, "--headless",
           f'--project={ilp}', f'--raw_data={raw_path}',
           f'--probabilities={pmap_path}', '--export_source=Multicut Segmentation',
           f'--output_filename_format={out_path}']
    subprocess.run(cmd)


def segment_all():
    for block_id in range(4):
        segment_mc(block_id)


if __name__ == "__main__":
    segment_all()
