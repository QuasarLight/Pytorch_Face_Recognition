from MegaFace_Evaluation.tools.plot_megaface_result import plot_megaface_result

if __name__ == '__main__':
    Evaluation_Results = ['/data/face_datasets/test_datasets/face_recognition/MegaFace/results/']

    Margin = ['ArcFace']

    probe = 'facescrub'

    other_methods_dir = None
    save_tpr_and_rank1_for_others = False
    target_fpr = 1e-6

    save_dir = './visualization_results'
    plot_megaface_result(Evaluation_Results, Margin,
                            probe,
                            save_dir,
                            other_methods_dir,
                            save_tpr_and_rank1_for_others,
                            target_fpr = target_fpr
                            )
