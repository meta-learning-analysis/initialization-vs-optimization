# initialization-vs-optimization


Implementation of **Meta-learning Initialization vs Optimizer: Beyond 20 ways Few-shot Learning** 

---
#### Abstract
Prevalent literature evaluates the merits of Meta-learning (ML) approaches on tasks with no more than 20 classes. We question this convention and deploy a more natural and complex task setup to test initialization and optimization ML approaches. Specifically, we expand the classes in the tasks to 200 for the Omniglot and 90 for tieredImagenet datasets. Interestingly, optimization approaches demonstrate higher resilience to complex tasks than initialization approaches, which is surprising as current task setups (5 way) advocate the superiority of initialization over the optimization approaches. We uncover intriguing patterns in the behavior of initialization and optimization approaches from different perspectives with increasing task complexity.  Our findings suggest that optimization approaches, despite learning a poor initialization, are better at adapting to newer tasks' support sets and often provide more confident predictions than initialization approaches. 

---- 

## Setup and Organization
The code repository is organized into the following sub-modules:
<ul>
<li> <b>algorithms</b>:
 Implements a generic metalearning framework (currently) supporting the algorithms - MAML, MetaSGD and MetaLSTM++ 
</li>

<li> <b>analysis</b>:
Contains the code for studying the properties and behaviours of the converged initialization and optimization based meta-learning models.
</li>

<li> <b>datasets</b>:
Implements few-shot learning task creation presently supporting the omniglot and tieredImagenet datasets. Adapted from the <a href="http://learn2learn.net"> learn2learn </a> library. 
</li>

<li> <b>models</b>:
 Contains the model definition for the Conv4 backbone (base model) and the learned optimizer (MetaLSTM++).
</li>

</ul>

Functionality of the remaining python scripts in the root directory:
<ul>

<li> <b>config.py</b>: 
 Implements the Config class that defines the configuration of the experiment to run. Set the member variables according to the experiment you wish to run. The key variables and their supported values are defined below:
        <ol>
            <li>
                <i> algo </i> : The meta-learning algorithm. Takes as value one of: ('MAML' , 'MetaSGD' , 'TA_LSTM'). TA_LSTM is  used as an alias for MetaLSTM++.
            </li>
            <li>
                <i> dataset </i> : The few-shot learning dataset. Takes as value one of: ('omniglot','tiered-imagenet')
            </li>
            <li>
                <i> n_class </i> : The no. of classes for the few shot learning task. Takes as input any int less than or equal to the total number of classes in the dataset considered.
            </li>
            <li>
                <i> n_shot </i> : The no. of shots per class for the few shot learning task. Takes as value any int. 
            </li>
            <li>
                <i> mode </i> :  The experiment mode. Takes as value one of: ('test', 'train').
            </li>
        </ol>
        
        
</p>
</li>

<li> <b>main.py</b>:
Used to train or test meta-learning algorithms defined using the Config class.
</li>

<li> <b>parser.py</b>:
Used to update experiment configuration from the command line. The configuration required should be passed as a dict after the <b>--arg</b> flag.
</li>
</ul>

The pretrained models and experiment files for all the settings can be found <a href="https://drive.google.com/drive/folders/1FV00_cb4Rk3uuVBOOYdSXLNjBreyiCj2?usp=sharing">here</a>. Download and place the<i>experiments</i> directory in the root directory to run analysis using them. 

-----

### Example commands

#### To train a meta-learning model (MAML) from scratch on the omniglot dataset in a 20-Way 1-Shot setting:

```python
python main.py --arg '{"algo":"MAML","dataset":"omniglot","n_class":20,"n_shot":1,"mode":"train","resume":0}'
```

#### To test a meta-learning model (MAML) from scratch on the omniglot dataset in a 20-Way 1-Shot setting:

```python
python main.py --arg '{"algo":"MAML","dataset":"omniglot","n_class":20,"n_shot":1,"mode":"test"}'
```

#### To run an analysis experiment (for eg. stress test):

```python
    python analysis/stress_test.py 
```

## How to cite

If you find this work useful in your research, kindly consider citing the following paper.

```python

```


## License

This project is distributed under [MIT license](LICENSE).

```c


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
