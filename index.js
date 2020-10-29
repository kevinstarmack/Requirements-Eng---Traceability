//can take in any number of FRs in "inputs"
//after opening cmd line
//type cd requirements_eng
//node index.js
//output found in Atom under "requirements-3nfr-80fr.txt"
const jaccard = require('jaccard');
const nlp = require('compromise');
const glob = require('glob');
const keywords = require('keyword-extractor');
const fs = require('fs')

const INPUTS_FOLDER = './inputs/';
const OUTPUTS_FOLDER = './outputs/';
const SIMILARITY_CUTOFF = 0.08;
// Get all inputs
glob.sync(`${INPUTS_FOLDER}*.txt`).forEach(path => {
    let nfr = [];
    let fr = [];

    // Read the file
    fs.readFile(path, 'utf-8', (err, contents) => {
        let lines = contents.split(/\r?\n/);

        lines.forEach(line => {
            if (line.length == 0) {
                return;
            }
            // Split the requirement text from the type (FR/NFR)
            let typeEnd = line.indexOf(':');
            let type = line.substr(0, typeEnd);
            let reqText = line.substr(typeEnd);

            let words = keywords.extract(reqText, {
                language: 'english',
                remove_digits: true,
                return_changed_case: true,
                remove_duplicates: false,
            });
            // Extract singular and infinite forms of word if possible
            words = words.map(word => {
                let noun = nlp(word).nouns().toSingular().text();
                let verb = nlp(word).verbs().toInfinitive().text();

                if (noun.length > 0) {
                    return noun;
                }
                if (verb.length > 0) {
                    return verb;
                }
                return word;
            });

            // Push to either NFR of FR req array
            let reqStorage = type.indexOf('NFR') != -1 ? nfr : fr;
            reqStorage.push({
                id: type,
                req: reqText,
                topics: words
            });
        });

        // Loop through FR and compute similarities with each NFR
        let outString = '';
        fr.forEach(frData => {
            let results = [];

            // Get the jaccard index of the NFR and FR keywords
            nfr.forEach(nfrData => {
                let index = jaccard.index(frData.topics, nfrData.topics);

                results.push(index);
            });

            // Based on the jaccard index and our cutoff, compute if the FR and NFR are related
            let r1 = results[0] > SIMILARITY_CUTOFF ? 1 : 0;
            let r2 = results[1] > SIMILARITY_CUTOFF ? 1 : 0;
            let r3 = results[2] > SIMILARITY_CUTOFF ? 1 : 0;

            outString += `${frData.id},${r1},${r2},${r3}\n`
        });

        // Get the filename
        let parts = path.split('/');
        let filename = parts[parts.length - 1];

        let outFilename = `trace-${filename}`;
        let outPath = `${OUTPUTS_FOLDER}${outFilename}`;

        // Write the entire result
        fs.writeFile(outPath, outString, (err) => {
            if (err) {
                console.error(err);
            } else {
                console.log(`Successfully saved output to ${outPath}`);
            }
        });
    });
});
