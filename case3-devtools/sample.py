import bioimageio.core
import napari


def segment_cells(image):
    nucleus_model = bioimageio.core.load_resource_description(
        "conscientious-seashell"
    )
    with bioimageio.create_prediction_pipeline(nucleus_model) as pp:
        nucleus_predictions = pp(image)
    nuclei = label(nucleus_predictions > 0.5)

    membrane_model = bioimageio.core.load_resource_description(
        "loyal-parrot"
    )
    with bioimageio.create_prediction_pipeline(membranes_model) as pp:
        membranes = pp(image)

    cells = watershed(membranes, seeds=nuclei)
    return cells


def classify_cells(image, cells):
    classification_model = bioimageio.core.load_resource_description(
        "straightforward-crocodile"
    )

    classes = []
    with bioimageio.create_prediction_pipeline(classification_model) as pp:
        for cell in cells:
            this_class = pp(image[cell.bounding_box])
            classes.append(this_class)
    return classes


def main(image):
    cells = segment_cells(image)
    classes = classify_cells(image, cells)
    visualize_in_napari(image, cells, classes)






visualize_hpa(
    "https://images.proteinatlas.org/10005/921_B9_1,Cytosol"
)






