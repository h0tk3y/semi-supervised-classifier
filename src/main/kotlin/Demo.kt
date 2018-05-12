
import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.JFreeChart
import org.jfree.chart.StandardChartTheme
import org.jfree.chart.plot.PlotOrientation
import org.jfree.chart.plot.XYPlot
import org.jfree.chart.renderer.xy.XYShapeRenderer
import org.jfree.chart.ui.ApplicationFrame
import org.jfree.chart.ui.RectangleInsets
import org.jfree.chart.ui.UIUtils
import org.jfree.chart.util.ShapeUtils
import org.jfree.data.xy.DefaultXYDataset
import org.jfree.data.xy.XYDataset
import java.awt.*
import javax.swing.JPanel

val nPerClass = 300
val labeledPart = nPerClass / 20

fun main(args: Array<String>) {
    val models = listOf(
        UncorrelatedGaussian(listOf(-10.0, 10.0), listOf(20.0, 10.0)),
        UncorrelatedGaussian(listOf(10.0, -10.0), listOf(100.0, 60.0)),
        UncorrelatedGaussian(listOf(0.0, 0.0), listOf(3.0, 10.0))
    )

    val estimator = UncorellatedGaussianEstimator()

    val classifier = MaxLikelihoodSemiSupervisedClassifier(estimator)

    val data = models.flatMap { m -> (1..nPerClass).map { m.sample() } }
    val labels = models.indices.flatMap { i ->
        (1..nPerClass).take(labeledPart).map { i } +
        (1..nPerClass).drop(labeledPart).map { null }
    }

    data.chunked(nPerClass).forEach {
        println("Chunk:")
        it.forEach { println(it.joinToString(", ")) }
    }

    val result = classifier.classify(data, labels)
    val labelsForClasses = result.chunked(nPerClass)
    println(labelsForClasses.map { classLabels -> classLabels.groupingBy { it }.eachCount() })

    println("-------")

    val classes = result.withIndex().groupBy { result[it.index] }.mapValues { it.value.map { data[it.index] } }
    for (c in classes) {
        println("Labeled:")
        c.value.forEach { println(it.joinToString(", ")) }
    }

    ScatterPlot.visualize(data.indices.map {
        DataItem(data[it], (it % nPerClass < labeledPart), it / nPerClass, result[it])
    })
}

data class DataItem(val vector: List<Double>, val given: Boolean, val originalClass: Int, val classifiedAs: Int)

class ScatterPlot(dataset: List<DataItem>) : ApplicationFrame("Data") {
    private val groups = dataset.groupBy { it.classifiedAs }.toSortedMap()

    init {
        contentPane = createDemoPanel().apply { preferredSize = Dimension(800, 600) }
    }

    init {
        ChartFactory.setChartTheme(StandardChartTheme("JFree/Shadow", true))
    }

    private fun createChart(dataset: XYDataset): JFreeChart {

        val chart = ChartFactory.createScatterPlot("", "", "", dataset, PlotOrientation.HORIZONTAL, false, true, false)

        val plot = chart.plot as XYPlot
        plot.domainGridlinePaint = Color.GRAY
        plot.rangeGridlinePaint = Color.GRAY
        plot.backgroundPaint = Color.DARK_GRAY
        plot.isOutlineVisible = false
        val renderer = plot.renderer

        plot.renderer = object : XYShapeRenderer() {
            override fun getItemShape(row: Int, column: Int): Shape {
                val dataItem = groups[row]!![column]
                val size = 4.0f
                return when (dataItem.originalClass) {
                    0 -> ShapeUtils.createDiagonalCross(size / 1.5f, size / 1.5f)
                    1 -> ShapeUtils.createRegularCross(size, size)
                    3 -> ShapeUtils.createDownTriangle(size)
                    else -> ShapeUtils.createUpTriangle(size)
                }
            }

            override fun getItemOutlinePaint(row: Int, column: Int): Paint = paint1(row, column, false)

            override fun getItemPaint(row: Int, column: Int): Paint {
                return paint1(row, column, true)
            }

            private fun paint1(row: Int, column: Int, shadedIfUnknown: Boolean): Paint {
                val dataItem = groups[row]!![column]
                val itemPaint = renderer.getItemPaint(row, column) as Color
                return if (!dataItem.given && shadedIfUnknown) Color(0, 0, 0, 0) else itemPaint
            }
        }.apply {
            drawOutlines = true
            defaultOutlineStroke = BasicStroke(1.0f)
        }

        return chart
    }

    /**
     * Creates a panel for the demo (used by SuperDemo.java).
     *
     * @return A panel.
     */
    fun createDemoPanel(): JPanel {
        val chart = createChart(createDataset())
        chart.padding = RectangleInsets(4.0, 8.0, 2.0, 2.0)
        val panel = ChartPanel(chart, false)
        panel.isMouseWheelEnabled = true
        panel.preferredSize = Dimension(600, 300)
        return panel
    }

    private fun createDataset(): XYDataset {
        val result = DefaultXYDataset()
        for ((k, v) in groups.mapValues { it.value.map { it.vector.toDoubleArray() } }) {
            result.addSeries(k, (0..1).map { i -> v.map { it[i] }.toDoubleArray() }.toTypedArray())
        }
        return result
    }

    companion object {
        fun visualize(data: List<DataItem>) {
            val demo = ScatterPlot(data)
            demo.pack()
            UIUtils.centerFrameOnScreen(demo)
            demo.isVisible = true
        }
    }
}
