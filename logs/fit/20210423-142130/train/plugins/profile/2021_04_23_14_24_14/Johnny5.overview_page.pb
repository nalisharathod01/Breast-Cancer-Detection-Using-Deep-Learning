?	v?+.???@v?+.???@!v?+.???@	L??v-T=?L??v-T=?!L??v-T=?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$v?+.???@? "5?b??AEg?EH??@Y?A??ފ??*	??SG??bA2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorf/??@!H#????X@)f/??@1H#????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?OVW??!???5?)?OVW??1???5?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism1_^?}t??! ?u?YA?)2?Lڔ?1}?;(Y+?:Preprocessing2F
Iterator::ModelQ???Ȭ?!?hD]??B?)2t??r?1??ln?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?'???@!^????X@)??9?l?1j??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9L??v-T=?IH?????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	? "5?b??? "5?b??!? "5?b??      ??!       "      ??!       *      ??!       2	Eg?EH??@Eg?EH??@!Eg?EH??@:      ??!       B      ??!       J	?A??ފ???A??ފ??!?A??ފ??R      ??!       Z	?A??ފ???A??ފ??!?A??ފ??b      ??!       JCPU_ONLYYL??v-T=?b qH?????X@Y      Y@q???{9?0?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 