import java.util.*
import kotlin.math.PI
import kotlin.math.pow

interface Model {
    fun likelihood(x: List<Double>): Double
    fun sample(): List<Double>
}

private val random = Random()

class UncorrelatedGaussian(
    val center: List<Double>,
    val variance: List<Double>
) : Model {
    init {
        require(center.size == variance.size)
    }

    val varianceProd = variance.reduce(Double::times)

    override fun likelihood(x: List<Double>): Double {
        val z = (x zip center).mapIndexed { i, (a, b) -> (a - b).pow(2) / variance[i] }.sum()
        val outOfExp = 1 / (2 * PI) / varianceProd
        val exp = Math.exp(-0.5 * z)
        return outOfExp * exp
    }

    override fun sample(): List<Double> = center.zip(variance) { x, d ->
        x + random.nextGaussian() * Math.sqrt(d)
    }
}

interface MaxLikelihoodEstimator<out T : Model> {
    fun estimate(data: List<List<Double>>): T
}

class UncorellatedGaussianEstimator : MaxLikelihoodEstimator<UncorrelatedGaussian> {
    override fun estimate(data: List<List<Double>>): UncorrelatedGaussian {
        val n = data.first().size
        require(data.all { it.size == n })

        val center = (0 until n).map { dim -> data.map { x -> x[dim] }.average() }
        val variance = (0 until n).map { dim -> data.map { x -> (x[dim] - center[dim]).pow(2) }.average() }

        return UncorrelatedGaussian(center, variance)
    }
}

fun main(args: Array<String>) {
    val g = UncorrelatedGaussian(listOf(10.0, 20.0), listOf(3.0, 4.0))
    val samples = (1..10000).map { g.sample() }
    val e = UncorellatedGaussianEstimator()
    val r = e.estimate(samples)
    println("${r.center}, ${r.variance}")
}