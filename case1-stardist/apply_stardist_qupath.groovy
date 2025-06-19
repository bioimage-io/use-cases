/**
 * Groovy script to apply Stardist using a model from bioimage.io.
 * For more information about StarDist + QuPath in general, see https://qupath.readthedocs.io/en/stable/docs/advanced/stardist.html
 *
 * This is written for compatibility with QuPath v0.5.1 and qupath-extension-stardist v0.5.0
 * It requires installing two extensions:
 *   - https://github.com/qupath/qupath-extension-stardist
 *   - https://github.com/qupath/qupath-extension-djl
 * It also only supports percentile normalization, with values that should be specified in the script 
 * (since the YAML is not parsed automatically).
 *
 * @author Pete Bankhead
 * @author Alan O'Callaghan
 */


import static qupath.lib.gui.scripting.QPEx.*
import org.bytedeco.opencv.opencv_core.Mat
import qupath.lib.gui.dialogs.Dialogs
import qupath.ext.stardist.StarDist2D
import qupath.opencv.tools.OpenCVTools

setPixelSizeMicrons(0.5, 0.5)

// specify the path to the model
var pathModel = '/home/alan/Documents/github/imaging/use-cases/case1-stardist/stardist-hne-nuclei-segmentation_tensorflow_saved_model_bundle'

String normalizationType = "percentile" // whether to use percentile normalization, alternatively "zeroMeanUnitVariance"
double minPercentile = 1.0
double maxPercentile = 99.8
boolean jointChannelNormalize = true            // Optionally normalize channels together (rather than independently)
double predictionThreshold = 0.5
double pixelSizeMicrons = 0.25             // Specify if input resolution should be set (rather than using full resolution)
boolean createAnnotations = true                // Create cells as annotations, which can then be edited (default is to create detection objects)
boolean includeProbability = true         // include the detection probability as an object measurement
double simplifyDistance = 0        // simplify distance threshold; set &le; 0 to turn off additional simplification
int tileSize = 1024


// define normalization parameters
var normalizerBuilder = StarDist2D.imageNormalizationBuilder()
    .maxDimension(4096)    // downsamples to ensure the image used to calculate normalization parameters are <= this value
    // .downsample(4) // alternatively, directly specify the downsample to use
    .useMask(true) // whether to mask the image using the object ROI when normalizing
    .perChannel(!jointChannelNormalize) // if joint normalization, then don't do per-channel normalization

if (normalizationType == "percentile") {
    println "Using percentile normalization"
    normalizerBuilder.percentiles(0.2, 99.8)  // Calculate image percentiles to use for normalization
} else if (normalizationType == "zeroMeanUnitVariance") {
    println "Using mean/variance normalization"
    normalizerBuilder.zeroMeanUnitVariance()
} else {
    println "Unknown normalization type; will use the default..."
}

// Customize how StarDist will operate
var stardistBuilder = StarDist2D.builder(pathModel)
        .threshold(predictionThreshold)
        .includeProbability(includeProbability)
        .simplify(simplifyDistance)
        .tileSize(tileSize, tileSize)
if (pixelSizeMicrons && Double.isFinite(pixelSizeMicrons))
    stardistBuilder.pixelSize(pixelSizeMicrons)

if (createAnnotations)
    stardistBuilder.createAnnotations()

stardistBuilder.preprocess(normalizerBuilder.build())

var stardist = stardistBuilder.build()

// Run detection for the selected objects
var imageData = getCurrentImageData()
var pathObjects = getSelectedObjects()
if (pathObjects.isEmpty()) {
    // If there are no annotations at all, and the image isn't too big, create an annotation for the full image
    if (getAnnotationObjects().isEmpty() && imageData.getServer().getWidth() <= 2048 && imageData.getServer().getHeight() <= 2048) {
        println "No objects are selected - I'll select the whole image then"
        createSelectAllObject(true)
    } else {
        Dialogs.showErrorMessage("StarDist", "Please select one or more parent objects!")
        return
    }
}
stardist.detectObjects(imageData, pathObjects)
stardist.close()
println 'Done!'

