package FraudDetection.SparkMLJob;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
/**
 *@author ekjot, hartaj, piyush
 *
 */
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import scala.Tuple2;

public class NewTrainModel {

	public static void main(String[] args) throws IOException {
		String mode = "cluster";
		if (args.length > 0 && args[0].equalsIgnoreCase("local"))
			mode = args[0];

		SparkConf sparkConf = new SparkConf();
		sparkConf.setAppName("Spark_ML_Job");
		if (mode.equals("local"))
			sparkConf.setMaster("local");

		sparkConf.set("spark.testing.memory", "2147480000");
		sparkConf.set("spark.driver.memory", "2g");

		JavaSparkContext javaSparkContext = null;
		JavaRDD<MyRow> dataset = null;
		JavaRDD<LabeledPoint> trainingData = null;
		JavaRDD<LabeledPoint> testData = null;

		RandomForestModel model = null;

		javaSparkContext = new JavaSparkContext(sparkConf);

		dataset = loadFile(javaSparkContext);
		System.out.println(dataset.count());
		JavaRDD<LabeledPoint>[] splits = splitDataSet(dataset);
		trainingData = splits[0];
		testData = splits[1];

		model = trainModel(trainingData);

		testModel(testData, model);

		// saveModel(javaSparkContext, model);
	}

	private static JavaRDD<MyRow> loadFile(JavaSparkContext javaSparkContext) {

		JavaRDD<MyRow> dataset = null;
		try {
			dataset = javaSparkContext.textFile(
					"F:\\Concor\\IDE\\ApacheSpark\\spark-2.4.4-bin-hadoop2.7\\bin\\src\\main\\java\\resources\\EkjotDataset\\preprocessed_dataset_without_testset.csv")
					.map(row -> {
						String[] arr = row.split(",");
						MyRow newRow = new MyRow();
						if (arr.length == 32) {

							newRow.setTime(Double.parseDouble(arr[1]));
							newRow.setAmount(Double.parseDouble(arr[30]));
							double[] V = new double[28];
							for (int i = 2; i < 30; i++) {
								V[i - 2] = Double.parseDouble(arr[i]);

							}
							newRow.setV(V);
							newRow.setClas(Double.parseDouble(arr[31]));
							// System.out.println("adding row");
							return newRow;
						}

						return null;
					});

		} catch (Exception e) {
			System.out.println(e);

		}

		return dataset;
	}

	private static void saveModel(JavaSparkContext javaSparkContext, RandomForestModel model) {
		String path = System.getProperty("user.dir") + "\\model\\" + "\\RandomForestClassificationModel\\";

		File f = new File(path);
		try {
			if (f.exists()) {
				FileUtils.forceDelete(f); // delete directory
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} // create directory

		model.save(javaSparkContext.sc(), path);
		System.out.println(path);
	}

	private static void testModel(JavaRDD<LabeledPoint> testData, RandomForestModel model) throws IOException {

		// Evaluate model on test instances and compute test error
		// JavaPairRDD<Object, Object> predictionAndLabel = testData.mapToPair(p -> {
		// if (p == null) {
		// return new Tuple2<>(-1.0, -1.0);
		// }
		// return new Tuple2<>(model.predict(p.features()), p.label());
		// });

		int numTrees = model.numTrees();
		DecisionTreeModel[] tm = model.trees();

		JavaPairRDD<Object, Object> probAndLabel = testData.mapToPair(p -> {
			if (p == null) {
				return new Tuple2<>(-1.0, -1.0);
			}
			return new Tuple2<>(

					// model.predict(p.features()),
					Arrays.stream(tm).mapToDouble(m -> {
						return m.predict(p.features());
					}).sum() / numTrees, p.label());
		});

		createFile(probAndLabel);

		//findError(testData, predictionAndLabel);

	}

	private static void createFile(JavaPairRDD<Object, Object> probAndLabel) throws IOException {
		// TODO Auto-generated method stub

		// JavaPairRDD<Object, Object> trueGenuine = probAndLabel.filter(pl -> ((double)pl._2==0.0));
		// JavaPairRDD<Object, Object> trueFraud = probAndLabel.filter(pl -> ((double)pl._2==1.0));

		// System.out.println("probandlabel");
		// print(probAndLabel);

		String str = "threshold,precision,recall" + "\n";
		for (double threshold = 0.0; threshold < 1.0; threshold += 0.1) {

			final double t = threshold;
			JavaPairRDD<Object, Object> genuine = probAndLabel.filter(pl -> ((double) pl._1 < t));
			JavaPairRDD<Object, Object> fraud = probAndLabel.filter(pl -> ((double) pl._1 >= t));

			// System.out.println("genuine"+threshold);
			// print(genuine);
			// System.out.println("fraud"+threshold);
			// print(fraud);

			double tp = fraud.filter(pl -> ((double) pl._2 == 1.0)).count();
			double fp = fraud.filter(pl -> ((double) pl._2 == 0.0)).count();
			// double tn= genuine.filter(pl -> ((double)pl._2==0.0)).count();
			double fn = genuine.filter(pl -> ((double) pl._2 == 1.0)).count();

			double precision = tp / (tp + fp);
			double recall = tp / (tp + fn);

			// System.out.println(tp+","+fp+","+fn);

			str += threshold + "," + precision + "," + recall + "\n";
		}

		BufferedWriter bw = new BufferedWriter(new FileWriter("DATASET_NAME.csv"));
		bw.write(str);
		bw.close();

	}

	private static void print(JavaPairRDD<Object, Object> probAndLabel) {
		// TODO Auto-generated method stub
		for (Tuple2<Object, Object> t : probAndLabel.collect()) {
			System.out.println(t._1 + "," + t._2);
		}
	}

	private static void findError(JavaRDD<LabeledPoint> testData, JavaPairRDD<Object, Object> predictionAndLabel) {
		// TODO Auto-generated method stub
		double testErr = predictionAndLabel.filter(pl -> (pl != null) && (!pl._1().equals(pl._2()))).count()
				/ (double) testData.filter(td -> td != null).count();
		System.out.println("Test Error: " + testErr);
		System.out.println("prediction and label:" + predictionAndLabel.collect() + ".....................");

		// Get evaluation metrics.
		BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(predictionAndLabel.rdd());

		System.out.println(otherMetrics(metrics));

	}

	private static String otherMetrics(BinaryClassificationMetrics metrics) {
		// TODO Auto-generated method stub

		String str = "";
		// Precision by threshold
		JavaRDD<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD();
		// System.out.println("Precision by threshold: " + precision.collect());
		str += "Precision by threshold: " + precision.collect() + "\n";

		// Recall by threshold
		JavaRDD<?> recall = metrics.recallByThreshold().toJavaRDD();
		// System.out.println("Recall by threshold: " + recall.collect());
		str += "Recall by threshold: " + recall.collect() + "\n";
		// F Score by threshold
		JavaRDD<?> f1Score = metrics.fMeasureByThreshold().toJavaRDD();
		// System.out.println("F1 Score by threshold: " + f1Score.collect());
		str += "F1 Score by threshold: " + f1Score.collect() + "\n";
		JavaRDD<?> f2Score = metrics.fMeasureByThreshold(2.0).toJavaRDD();
		// System.out.println("F2 Score by threshold: " + f2Score.collect());
		str += "F2 Score by threshold: " + f2Score.collect() + "\n";
		// Precision-recall curve
		JavaRDD<?> prc = metrics.pr().toJavaRDD();
		// System.out.println("Precision-recall curve: " + prc.collect());
		str += "Precision-recall curve: " + prc.collect()
		+ "\n...........................................................";
		// Thresholds
		JavaRDD<Double> thresholds = precision.map(t -> Double.parseDouble(t._1().toString()));
		str += "thresholds : " + thresholds.collect() + "..........................................\n";

		// ROC Curve
		JavaRDD<?> roc = metrics.roc().toJavaRDD();
		// System.out.println("ROC curve: " + roc.collect());
		str += "ROC curve: " + roc.collect() + "\n";

		// AUROC
		// System.out.println("Area under ROC = " + metrics.areaUnderROC());
		str += "Area under ROC = " + metrics.areaUnderROC() + "\n";

		// AUPRC
		// System.out.println("Area under precision-recall curve = "
		// +metrics.areaUnderPR());
		str += "Area under precision-recall curve = " + metrics.areaUnderPR() + "\n";
		return str;
	}

	private static RandomForestModel trainModel(JavaRDD<LabeledPoint> trainingData) {
		// Train a RandomForest model.
		// Empty categoricalFeaturesInfo indicates all features are continuous.
		int numClasses = 2;
		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
		int numTrees = 300;
		String featureSubsetStrategy = "auto"; // Let the algorithm choose.
		String impurity = "gini";
		int maxDepth = 5;
		int maxBins = 32;
		int seed = 12345;
		return RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, numTrees,
				featureSubsetStrategy, impurity, maxDepth, maxBins, seed);
	}

	private static JavaRDD<LabeledPoint>[] splitDataSet(JavaRDD<MyRow> dataset) {
		JavaRDD<LabeledPoint> labeledDataSet = dataset.map((MyRow myRow) -> {
			return new LabeledPoint(myRow.getClas(), myRow.getRowVector());
		});
		return labeledDataSet.randomSplit(new double[] { 0.8, 0.2 }, 11L);

	}

}
