for ((year=2022; year>=2017; year--)); do
  echo ${year}
  python analyze.py -y $year -rw --model gpt --inflation_type cause
#   python3 "${python_file}" "${year}"
done