# import all estimators that we have into here
import covariance_estimators
import estimators_testing

covariance_matrix_estimators = {
    "cov1para": covariance_estimators.cov1Para,
    "sample": covariance_estimators.sample_covmat,
    "cov2para": estimators_testing.cov2Para,
    "cov_diag": estimators_testing.covDiag,
    "cov_cor": estimators_testing.covCor,
    "QIS": estimators_testing.QIS,

}

params = {
    "end_date": 20171231,
    "estimation_window_length": 1,
    "out_of_sample_period_length": 20,
    "pf_size": [100],
    "estimator": covariance_matrix_estimators,
    "raw_data_path": None,
    "return_data_path": r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\return_matrices",
    "result_data_path": r"C:\Users\Damja\OneDrive\Damjan\FS23\master-thesis\code\results",
}
