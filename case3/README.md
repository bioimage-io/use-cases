# Use-case 3: Classification, imjoy & python library usage

## The model

We use 10.5281/zenodo.5910855 (not on the website yet, PR: https://github.com/bioimage-io/collection-bioimage-io/pull/266).

## Usage in python library

The `hpa_app.py` loads data from the hpa website, runs cell segmentation and then classifies each cell.

TODO/ISSUES:
- Wei will follow up if we need to resize the image.
- Wei will also look into wrapping the original segmentation model.
- The current class predictions don't look right, might be due to not resizeing or some issue in the channel order.

The original segmentation functionality:
https://github.com/CellProfiling/HPA-Cell-Segmentation/blob/master/hpacellseg/cellsegmentator.py
https://github.com/CellProfiling/HPA-Cell-Segmentation/blob/master/hpacellseg/utils.py


## Usage in imjoy / bioengine

TODO

## Usage in deepimagej

TODO
