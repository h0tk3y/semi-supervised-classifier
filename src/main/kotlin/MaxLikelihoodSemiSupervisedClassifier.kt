class MaxLikelihoodSemiSupervisedClassifier<out T : Model>(
    val estimator: MaxLikelihoodEstimator<T>
) {
    fun classify(data: List<List<Double>>, labels: List<Int?>): List<Int> {
        require(data.size == labels.size)

        val classIndices = data.indices
            .filter { labels[it] != null }
            .groupBy { labels[it]!! }
            .mapValues { it.value.toMutableList() }

        val learnedLabels = labels.toMutableList()

        lateinit var classModels: List<Pair<Int, T>>

        while (classIndices.values.sumBy { it.size } < data.size) {
            classModels = classIndices.entries.map { (label, indices) ->
                val dataItems = data.slice(indices)
                label to estimator.estimate(dataItems)
            }

            val unlabeledIndices = data.indices.filter { learnedLabels[it] == null }
            val unlabeledClassLikelihood = classModels.map { (label, model) ->
                label to unlabeledIndices.map { i -> model.likelihood(data[i]) }
            }

            val argMax = unlabeledIndices.indices.map { i ->
                unlabeledClassLikelihood
                    .maxBy { (_, likelihood) -> likelihood[i] }!!
                    .let { (label, _) -> label }
            }

            val margin = argMax.mapIndexed { i, y ->
                val maxL = unlabeledClassLikelihood[y].second[i]
                unlabeledClassLikelihood.mapNotNull { (l, likelihood) ->
                    val diff = maxL - likelihood[i]
                    diff.takeIf { y != l }
                }.min() ?: 0.0
            }

            val marginLimitByTop = margin.max()!! * 0.5

            val newLabels = unlabeledIndices
                .mapIndexedNotNull { index, i -> (i to argMax[index]).takeIf { margin[index] >= marginLimitByTop } }

            println("Learned ${newLabels.size} labels ")

            newLabels.forEach { (index, toClass) ->
                learnedLabels[index] = toClass
                classIndices[toClass]!!.add(index)
            }
        }

        val result = learnedLabels.map { requireNotNull(it) { "All items should be classified." } }
        return result
    }
}