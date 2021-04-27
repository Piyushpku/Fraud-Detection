package FraudDetection.SparkPredictApp;

import java.io.Serializable;

import org.apache.spark.mllib.linalg.DenseVector;

public class MyRow implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private double time;
	private double clas;
	private double amount;
	private double[] V = new double[28];

	/**
	 * @return the time
	 */
	public double getTime() {
		return time;
	}

	/**
	 * @param time the time to set
	 */
	public void setTime(double time) {
		this.time = time;
	}

	/**
	 * @return the clas
	 */
	public double getClas() {
		return clas;
	}

	/**
	 * @param clas the clas to set
	 */
	public void setClas(double clas) {
		this.clas = clas;
	}

	/**
	 * @return the amount
	 */
	public double getAmount() {
		return amount;
	}

	/**
	 * @param amount the amount to set
	 */
	public void setAmount(double amount) {
		this.amount = amount;
	}

	/**
	 * @return the v
	 */
	public double[] getV() {
		return V;
	}

	/**
	 * @param v the v to set
	 */
	public void setV(double[] v) {
		if (V.length != 28) {
			System.out.println("Passed Array Is Invalid.");
			return;
		}
		V = v;
	}

	public String getVInString() {
		String str = "";
		for (int i = 0; i < 28; i++) {
			str += V[i] + ", ";
		}
		str = str.substring(0, str.length() - 2);
		return str;
	}

	public DenseVector getRowVector() {

		double[] rowVect = new double[30];

		rowVect[0] = time;
		rowVect[1] = amount;

		for (int i = 2; i < 30; i++) {
			rowVect[i] = V[i - 2];
		}

		return (new DenseVector(rowVect));

	}
}
