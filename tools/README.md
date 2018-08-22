## Usage

### Plot

```
python plot.py --data [DATA PATHS]
```

Arguments:

 * `--data`: Paths that data are saved. It can be assigned by multiple paths.
 * `--name`: Names of the data. It can be assigned by multiple names. (default: None)
 * `--title`: Figure title. (default: `'Training Curve'`)
 * `--xlabel`: Description of x-coordinate. (default: `'Epochs'`)
 * `--ylabel`: Description of y-coordinate. (default: `'Accuracy'`)
 * `--figure_num`: Figure ID. (default: None)
 * `--save`: A flag used to decide whether to save the figure or not.
 * `--display`: A flag used to decide whether to display the figure or not. 
 * `--filename`: Name of the saved figure. (default: `'figure.png'`)

### Average Records

```
python average.py -i [INPUT_FILE] -n [NUM] -o [OUTPUT]
```

Arguments:

The program will read the data from the files `'INPUT_FILE{}.json'` where `{}` is the number from `0` to `NUM-1`.  
Then, the program will average the result and output to file `OUTPUT`.  
For example, executing
```
python average.py -i record -n 3 -o output.json
```
will average the data from `record0.json`, `record1.json` and `record2.json` and output the averaged result to `output.json`.
