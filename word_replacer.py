

filenames=['Am93.com', 'Am95.com', 'Am72.com', 'Am85.com', 'Am86.com', 'Am70.com', 'Am97.com', 'Am77.com', 'Am87.com', 'Am90.com', 'Am75.com', 'Am74.com', 'Am102.com', 'Am68.com', 'Am145.com', 'Am99.com', 'Am94.com', 'Am164.com', 'Am96.com', 'Am98.com', 'Am91.com', 'Am104.com', 'Am84.com', 'Am101.com', 'Am69.com', 'Am83.com', 'Am92.com', 'Am71.com', 'Am103.com', 'Am73.com', 'Am82.com', 'Am76.com', 'Am100.com']

def string_replacer(file, old, new):
    reading_file=open(file, 'r')
    new_file_content=''
    for line in reading_file:
        stripped_line=line.strip()
        new_line=stripped_line.replace(old, new)
        new_file_content+=new_line + "\n"
    reading_file.close()
    
    writing_file=open(file, 'w')
    writing_file.write(new_file_content)
    writing_file.close()





if __name__ == "__main__":
	for file in filenames:
		 string_replacer(file, 'tert-Butyl alcohol', '2-Methyl-2-Propanol')




