#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#include <boost/regex.hpp>
#include <H5Cpp.h>

typedef std::vector<bool> haplotype;
typedef std::vector<haplotype> matrix;

void getGenotypes(std::string str, bool *snpmat, int n, int m, int snpcounter) {
    // Find genotype chunks with regex:
    // each section start with the chromosome number.
    // We find the first genotype and then loop through 
    // others up to n genotypes.
    boost::regex reg("\\d{1}\\|\\d{1}");
    boost::match_results<std::string::const_iterator> match;
    std::string::const_iterator strini = str.begin();
    // std::string::const_iterator strend = str.end();
    // If a genotype is found we find its position in the string
    // and search for n more genotypes.
    if (boost::regex_search(str, match, reg, boost::match_extra)) { 
        int matchini = ((strini - str.begin()) + match.position());
        int idvcounter = 0;
        for (unsigned int i = 0; i < n/2; i++) {  
            std::string genotype = str.substr(matchini, 3);
            bool maternalSNP = (genotype[0] != '0');
            bool paternalSNP = (genotype[2] != '0');
            snpmat[m*idvcounter+snpcounter] = maternalSNP;
            snpmat[m*(idvcounter+1)+snpcounter] = paternalSNP;
            matchini += 4;
            idvcounter += 2;
        }
    } else {
        std::cout << "No matches..." << std::endl;
    }
}

void VCF2SNP(std::string fileName, int chr, bool *snpmat, std::string *ids, int n, int m) {
    // Open the VCF file.
    std::ifstream in(fileName.c_str());
    // Check whether the object is valid.
    if (!in) {
        std::cerr << "Cannot open the file: "<< fileName <<std::endl;
        throw; 
    }
    std::string str;
    int snpcounter = 0;
    // Read the next line from file until it reaches
    // the desired number of SNPs, m.
    while (getline(in, str)) {
        std::istringstream iss(str);
        int hchr; // Skip line if does not start with chr.
        if (!(iss >> hchr) || hchr != chr) { 
            //std::cout << "Line" << counter << ": Skip" << std::endl;
            if (str.substr(0, 6) == "#CHROM") {
                std::stringstream ss(str);
                int i = 0;
                std::set<std::string> header = {"#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"};
                bool inHeader = true;
                for (auto w = std::istream_iterator<std::string>(ss); w != std::istream_iterator<std::string>() && i < n; w++) {
                    if (inHeader && header.count(*w)) {
                        // Ignore:
                        // std::cout << "Skipping " << *w << std::endl;
                    } else {
                        inHeader = false;
                        ids[i] = *w;
                        i++;
                    }
                }
            }
        } else {
            // Read the genotypes for SNP at snpcounter position.
            getGenotypes(str, snpmat, n, m, snpcounter);  
            snpcounter++; 
        }
        if (snpcounter == m) break;
    }
    // Close the file.
    in.close();
}

void SNP2HDF5(std::string filename, bool *snpmat, std::string *ids, int n, int m) {
    // HDF5 only understands vector of char* :-(
    std::vector<const char*> arr_c_str;
    for (size_t i = 0; i < n; ++i) {
        arr_c_str.push_back(ids[i].c_str());
    }
    std::cout << "Saving HDF5 ..." << std::endl;
    
    H5::H5File *file = new H5::H5File(filename, H5F_ACC_TRUNC);
    hsize_t datadim[] = { static_cast<unsigned long long>(n), static_cast<unsigned long long>(m) };
    hsize_t datadimIds[] = { static_cast<unsigned long long>(n) };
    // Create dataspace for the dataset in the file. Rank is 2.
    H5::DataSpace dataspace(2, datadim);
    H5::DataSpace dataspaceIds(1, datadimIds);
    H5::DataType datatype(H5::PredType::NATIVE_HBOOL);
    H5::StrType datatypeIds(H5::PredType::C_S1, H5T_VARIABLE);
    // Create dataset and write it into the file.
    H5::DataSet *dataset = new H5::DataSet(file->createDataSet("snps", datatype, dataspace));
    H5::DataSet *datasetIds = new H5::DataSet(file->createDataSet("labels", datatypeIds, dataspaceIds));
    //std::cout << "Name: "<< dataset->getFileName() << "." << std::endl;
    try {
        dataset->write(snpmat, datatype);
        datasetIds->write(arr_c_str.data(), datatypeIds);
    }
    catch (int e) {
        std::cout << "An exception occurred. Exception #" << e << '\n';
    }
    // Clear memory.
    delete dataset;
    delete datasetIds;
    delete file;
    dataset = NULL;
    datasetIds = NULL;
    file = NULL;
    std::cout << "Data stored in "<< filename << "." << std::endl;
}

int main(int argc, char **argv) { 
    // argv[1] = chr num;
    // argv[2] = n dimension;
    // argv[3] = m dimension;
    // argv[4] = VCF file name;
    // argv[5] = HDF5 out file name;
    if (argv[1] == NULL || argv[2] == NULL || argv[3] == NULL | argv[4] == NULL){
        std::cout << "Missing dimensions." << std::endl;
        exit(1);
    }
    int chr;
    long long n, m;
    chr = std::atoi(argv[1]);
    n = std::atoi(argv[2])*2;
    m = std::atoi(argv[3]);
    std::string file = argv[4];
    std::string out;
    if (argv[5] == NULL) {
        std::stringstream ss;
        ss << "chr" << chr << "SNPdata" << n << "x" << m << ".h5";
        out = ss.str();
    } else {
        out = argv[5];
    }
    // Define dynamic array for SNP storage.
    std::cout << "Allocating an array of size " << (n*m*sizeof(bool)>>20) << " MB." << std::endl;
    bool *snpmat = new bool[n*m];
    std::string *ids = new std::string[n];
    // Read SNP data from VCF file into bool array.
    VCF2SNP(file, chr, snpmat, ids, n, m);
    // Store the bool array into HDF5 file.
    SNP2HDF5(out, snpmat, ids, n, m);
    // Clear memory.
    delete[] snpmat;
    delete[] ids;
    snpmat = NULL;
    ids = NULL;
    return 0;
}
