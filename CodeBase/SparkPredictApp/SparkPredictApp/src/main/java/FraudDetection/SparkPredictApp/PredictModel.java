/**
 *
 */
package FraudDetection.SparkPredictApp;

import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.border.TitledBorder;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import scala.Tuple2;

/**
 * @author ekjot
 *
 */
public class PredictModel {

	private static JButton predictButton;
	private static JButton loadModelButton;
	private static JTextField modelAddressTextField;
	private static JTextField fileAddressTextField;
	private static JButton loadFileButton;
	private static JTextArea resultTextArea;
	private static JPanel modelPanel;
	private static JLabel modelAddressLabel;
	private static JPanel filePanel;
	private static JLabel fileAddressLabel;
	private static JPanel predictPanel;
	private static JScrollPane scrollPane;
	private static RandomForestModel model;
	private static JavaSparkContext javaSparkContext;
	private static JavaRDD<LabeledPoint> labeledDataSet;

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		JFrame frame = initialize();

		frame.setVisible(true);

		SparkConf sparkConf = new SparkConf();
		sparkConf.setAppName("Spark_ML_Job");
		sparkConf.setMaster("local");
		javaSparkContext = new JavaSparkContext(sparkConf);
	}

	private static JFrame initialize() {
		JFrame frame;
		Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		frame = new JFrame("SparkPredictApp");
		frame.setBounds(screenSize.width / 4, screenSize.height / 4, screenSize.width / 3, screenSize.height / 2);
		frame.setResizable(false);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().setLayout(new FlowLayout());

		modelPanel = new JPanel();
		modelPanel.setBorder((TitledBorder) BorderFactory.createTitledBorder("Model"));

		modelAddressLabel = new JLabel("Model Address:  ");
		modelPanel.add(modelAddressLabel);

		modelAddressTextField = new JTextField("", 20);
		modelAddressTextField.setText("");
		modelPanel.add(modelAddressTextField);

		loadModelButton = new JButton("Load Model");
		loadModelButton.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				loadModelButtonPressed();

			}
		});
		modelPanel.add(loadModelButton);

		filePanel = new JPanel();
		filePanel.setBorder((TitledBorder) BorderFactory.createTitledBorder("File"));

		fileAddressLabel = new JLabel("File Address(*.txt):  ");
		filePanel.add(fileAddressLabel);

		fileAddressTextField = new JTextField("", 20);
		fileAddressTextField.setText("");
		filePanel.add(fileAddressTextField);

		loadFileButton = new JButton("Load File");
		loadFileButton.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				loadFileButtonPressed();

			}
		});

		filePanel.add(loadFileButton);

		predictPanel = new JPanel();
		predictPanel.setBorder((TitledBorder) BorderFactory.createTitledBorder("Predict"));

		predictButton = new JButton("Predict");
		predictButton.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent arg0) {
				predictButtonPressed();
			}
		});
		predictPanel.add(predictButton);

		resultTextArea = new JTextArea(10, 31);
		resultTextArea.setEditable(false);

		scrollPane = new JScrollPane();
		scrollPane.setViewportView(resultTextArea);
		predictPanel.add(scrollPane);

		frame.getContentPane().add(modelPanel);
		frame.getContentPane().add(filePanel);
		frame.getContentPane().add(predictPanel);

		return frame;
	}

	protected static void loadFileButtonPressed() {
		String path = fileAddressTextField.getText().trim();
		if (!path.equals("")) {
			loadFile(path);
		}
	}

	private static void loadFile(String path) {

		JavaRDD<MyRow> dataset = null;
		try {
			//System.out.println("Fuddu Code aagya...file read krna shuru");

			dataset = javaSparkContext.textFile(path).map(row -> {

				String[] arr = row.split(",");
				//System.out.println("Fuddu Code aagya..."+arr.length);
				MyRow newRow = new MyRow();
				if (arr.length == 31) {
					//System.out.println("Fuddu Code aagya...");
					newRow.setTime(Double.parseDouble(arr[0]));
					newRow.setAmount(Double.parseDouble(arr[1]));
					double[] V = new double[28];
					for (int i = 2; i < 30; i++) {
						V[i - 2] = Double.parseDouble(arr[i]);

					}
					newRow.setV(V);
					newRow.setClas(Double.parseDouble(arr[30]));
					return newRow;
				}
				return null;
			});

			//System.out.println("fuddu kammmmmmmmm"+dataset.count());

			JOptionPane.showMessageDialog(loadFileButton, "File Successfully Loaded.");
		} catch (Exception e) {
			System.out.println(e);
			JOptionPane.showMessageDialog(loadFileButton, "File Cannot Be Loaded.");
		}

		labeledDataSet = dataset.map(myRow -> {
			if (myRow != null) {
				return new LabeledPoint(myRow.getClas(), myRow.getRowVector());
			}
			return null;
		});

	}

	private static void loadModel(String path) {
		try {
			path = "file:/" + path;
			model = RandomForestModel.load(javaSparkContext.sc(), path);
			JOptionPane.showMessageDialog(loadModelButton, "Model Successfully Loaded.");

		} catch (Exception e) {
			System.out.println(e);
			JOptionPane.showMessageDialog(loadModelButton, "Model Not Found!!!");
		}

	}

	protected static void loadModelButtonPressed() {
		String path = modelAddressTextField.getText().trim();
		if (!path.equals("")) {
			loadModel(path);
		}

	}

	protected static void predictButtonPressed() {
		JavaPairRDD<Object, Object> predictions = labeledDataSet.mapToPair(p -> {
			if (p != null) {
				//System.out.println(model.predict(p.features()) + p.label());
				return new Tuple2<>(model.predict(p.features()), p.label());
			}
			return new Tuple2<>(-1.0, -2.0);
		});

		resultTextArea.setText("");
		int i = 1;
		Double tp = 0.0, fp = 0.0, tn = 0.0, fn = 0.0;
		for (Tuple2 t : predictions.collect()) {
			Double d = (Double) t._1;
			Double act = (Double) t._2;
			if (d != -1.0) {
				System.out.println("* " + d);
				resultTextArea.append("Transaction " + i++ + ": " + ((d == 1.0) ? "Fraud!!!" : "Genuine") + "\n");
				//predict its fraud
				if (d == 1.0 && act == 1.00) {
					tp++;
				} else if (d == 1.0 && act == 0.00) {
					fp++;
				} else if (d == 0.0 && act == 0.00) {
					tn++;
				} else if (d == 0.0 && act == 1.00) {
					fn++;
				}
			}
		}
		resultTextArea.append("tp=" + tp + "  fp=" + fp + "  tn=" + tn + "  fn=" + fn + "\n");
		resultTextArea.append("Precision is: " + tp / (tp + fp) + "\n");
		resultTextArea.append("Recall is: " + tp / (tp + fn) + "\n");
		findError(labeledDataSet, predictions);

	}

	private static void findError(JavaRDD<LabeledPoint> testData, JavaPairRDD<Object, Object> predictions) {
		// TODO Auto-generated method stub
		double testErr = predictions.filter(pl -> (pl != null) && (!pl._1().equals(pl._2()))).count()
				/ (double) testData.filter(td -> td != null).count();
		System.out.println("Test Error: " + testErr);

		// Get evaluation metrics.
		BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(predictions.rdd());

		resultTextArea.append(otherMetrics(metrics));

	}

	private static String otherMetrics(BinaryClassificationMetrics metrics) {
		// TODO Auto-generated method stub

		String str = "";
		// // Precision by threshold
		// JavaRDD<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD();
		// // System.out.println("Precision by threshold: " + precision.collect());
		// str += "Precision by threshold: " + precision.collect() + "\n";
		//
		// // Recall by threshold
		// JavaRDD<?> recall = metrics.recallByThreshold().toJavaRDD();
		// // System.out.println("Recall by threshold: " + recall.collect());
		// str += "Recall by threshold: " + recall.collect() + "\n";
		// // F Score by threshold
		// JavaRDD<?> f1Score = metrics.fMeasureByThreshold().toJavaRDD();
		// // System.out.println("F1 Score by threshold: " + f1Score.collect());
		// str += "F1 Score by threshold: " + f1Score.collect() + "\n";
		// JavaRDD<?> f2Score = metrics.fMeasureByThreshold(2.0).toJavaRDD();
		// // System.out.println("F2 Score by threshold: " + f2Score.collect());
		// str += "F2 Score by threshold: " + f2Score.collect() + "\n";
		// // Precision-recall curve
		// JavaRDD<?> prc = metrics.pr().toJavaRDD();
		// // System.out.println("Precision-recall curve: " + prc.collect());
		// str += "Precision-recall curve: " + prc.collect() + "\n";
		// // Thresholds
		// JavaRDD<Double> thresholds = precision.map(t -> Double.parseDouble(t._1().toString()));
		//
		// // ROC Curve
		// JavaRDD<?> roc = metrics.roc().toJavaRDD();
		// // System.out.println("ROC curve: " + roc.collect());
		// str += "ROC curve: " + roc.collect() + "\n";
		//
		// // AUROC
		// // System.out.println("Area under ROC = " + metrics.areaUnderROC());
		// str += "Area under ROC = " + metrics.areaUnderROC() + "\n";

		// AUPRC
		// System.out.println("Area under precision-recall curve = " +metrics.areaUnderPR());
		str += "Area under precision-recall curve = " + metrics.areaUnderPR() + "\n";
		return str;
	}

}