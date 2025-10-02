python evaluate_dataset.py --transcript_file transcribed_data/whisper_commonvoice_test_en.parquet --beam_size 100 --save_results
python evaluate_dataset.py --transcript_file transcribed_data/phi_commonvoice_test_en.parquet --beam_size 100 --save_results
python evaluate_dataset.py --transcript_file transcribed_data/parakeet-tdt_commonvoice_test_en.parquet --beam_size 100 --save_results
python evaluate_dataset.py --transcript_file transcribed_data/parakeet-ctc_commonvoice_test_en.parquet --beam_size 100 --save_results

python evaluate_dataset.py --transcript_file transcribed_data/whisper_tedlium_test_en.parquet --beam_size 100 --save_results
python evaluate_dataset.py --transcript_file transcribed_data/phi_tedlium_test_en.parquet --beam_size 100 --save_results
python evaluate_dataset.py --transcript_file transcribed_data/parakeet-tdt_tedlium_test_en.parquet --beam_size 100 --save_results
python evaluate_dataset.py --transcript_file transcribed_data/parakeet-ctc_tedlium_test_en.parquet --beam_size 100 --save_results

python evaluate_dataset.py --transcript_file transcribed_data/whisper_primock57_test_en.parquet --beam_size 100 --save_results
python evaluate_dataset.py --transcript_file transcribed_data/phi_primock57_test_en.parquet --beam_size 100 --save_results
python evaluate_dataset.py --transcript_file transcribed_data/parakeet-tdt_primock57_test_en.parquet --beam_size 100 --save_results
python evaluate_dataset.py --transcript_file transcribed_data/parakeet-ctc_primock57_test_en.parquet --beam_size 100 --save_results