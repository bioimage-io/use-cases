/**
 * Groovy script to apply Stardist using a model from bioimage.io.
 * For more information about StarDist + QuPath in general, see https://qupath.readthedocs.io/en/stable/docs/advanced/stardist.html
 *
 * Note that this is written for compatibility with QuPath v0.3.2... and is therefore rather awkward.
 * It requires separately installing two extensions:
 *   - https://github.com/qupath/qupath-extension-stardist
 *   - https://github.com/qupath/qupath-extension-tensorflow
 * It also only supports percentile normalization, with values that should be specified in the script 
 * (since the YAML is not parsed automatically).
 *
 * Future QuPath versions will reduce the complexity a bit by including more useful built-in methods.
 *
 * @author Pete Bankhead
 */


import static qupath.lib.gui.scripting.QPEx.*
import org.bytedeco.opencv.opencv_core.Mat
import qupath.lib.gui.dialogs.Dialogs
import qupath.ext.stardist.StarDist2D
import qupath.opencv.tools.OpenCVTools

setPixelSizeMicrons(0.5, 0.5)

// specify the path to the model
var pathModel = '/home/pape/Work/bioimageio/use-cases/case2/he-model-pretrained/bioimageio'
// Define nput normalization percentiles etc
double minPercentile = 1.0                       // Input normalization min percentile
double maxPercentile = 99.8                      // Input normalization max percentile
boolean jointChannelNormalize = true            // Optionally normalize channels together (rather than independently)
double predictionThreshold = 0.5  // Prediction threshold
double pixelSizeMicrons = 0.25             // Specify if input resolution should be set (rather than using full resolution)
boolean createAnnotations = true                // Create cells as annotations, which can then be edited (default is to create detection objects)

// Customize how StarDist will operate
var stardistBuilder = StarDist2D.builder(pathModel)
        .threshold(predictionThreshold)
if (pixelSizeMicrons && Double.isFinite(pixelSizeMicrons))
    stardistBuilder.pixelSize(pixelSizeMicrons)

if (createAnnotations)
    stardistBuilder.createAnnotations()

if (jointChannelNormalize)
    stardistBuilder.preprocess(
            new NormalizePercentileJointOp(minPercentile, maxPercentile)
    )
else
    stardistBuilder.normalizePercentiles(minPercentile, maxPercentile)

// If necessary, customize tile size
//stardistBuilder.tileSize(1024, 1024)

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
println 'Done!'



/**
 * For QuPath v0.3.2, we need a custom ImageOp to perform joint channel normalization
 */
class NormalizePercentileJointOp implements qupath.opencv.ops.ImageOp {

    private double[] percentiles

    NormalizePercentileJointOp(double percentileMin, double percentileMax) {
        this.percentiles = [percentileMin, percentileMax] as double[]
        if (percentileMin == percentileMax)
            throw new IllegalArgumentException("Percentile min and max values cannot be identical!")
    }

    @Override
    public Mat apply(Mat mat) {
        // Need to reshape for percentiles to be correct in QuPath v0.3.2
        var matTemp = mat.reshape(1, mat.rows()*mat.cols())
        var range = OpenCVTools.percentiles(matTemp, percentiles)
        double scale
        if (range[1] == range[0]) {
            println("Normalization percentiles give the same value ({}), scale will be Infinity", range[0])
            scale = Double.POSITIVE_INFINITY
        } else
            scale = 1.0/(range[1] - range[0])
        double offset = -range[0]
        mat.convertTo(mat, mat.type(), scale, offset*scale)
        return mat;
    }

}
