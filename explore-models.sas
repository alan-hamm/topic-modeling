
libname A postgres server='127.0.0.1' port=5432  user=postgres password='admin' database=lda_manager_test;


proc import datafile="C:\_harvester\data\lda-models\2020-2024\metadata\metadata-09282024.csv"
	out=metadata
	dbms=dlm
	replace;
	delimiter=';';
	guessingrows=max;
run;


/*
expected_dtypes = {
        'type': str,
        'batch_size': int,
        'text': object,  # Use object dtype for lists of strings (file paths)
        'text_sha256': str,
        'text_md5': str,
        'convergence': 'float32',
        'perplexity': 'float32',
        'coherence': 'float32',
        'topics': int,
        # Use pd.Categorical.dtype for categorical columns
        # Ensure alpha and beta are already categorical when passed into this function
        # They should not be wrapped again with CategoricalDtype here.
        'alpha_str': str,
        'n_alpha': 'float32',
        'beta_str': str,
        'n_beta': 'float32',
        'passes': int,
        'iterations': int,
        'update_every': int,
        'eval_every': int,
        'chunksize': int,
        'random_state': int,
        'per_word_topics': bool,
        'top_words': object,
        'lda_model': object,
        # Enforce datetime type for time
        'time': 'datetime64[ns]',
    } 
*/


data WORK.METADATA (drop=_index)  ;
%let _EFIERR_ = 0; /* set the ERROR detection macro variable */
infile "C:\_harvester\data\lda-models\2020-2024\metadata\metadata-09282024.csv" delimiter = ';' missover DSD lrecl=32767 firstobs=2 TERMSTR=cr;
	length index 3;
   informat _index $84. ;
   informat time_key $1024.;
   informat type $53. ;
   informat num_workers 3.;
   informat batch_size best32. ;
   informat num_documents 3.;
   informat text_sha256 $64. ;
   informat text_md5 $32. ;
   informat convergence best32. ;
   informat perplexity best32. ;
   informat coherence best32. ;
   informat topics best32. ;
   informat alpha_str $18. ;
   informat n_alpha best32. ;
   informat beta_str $18. ;
   informat n_beta best32. ;
    informat passes best32. ;
    informat iterations best32. ;
    informat update_every best32. ;
    informat eval_every best32. ;
    informat chunksize best32. ;
    informat random_state best32. ;
    informat per_word_topics $5. ;
    informat top_words $2048. ;
	informat create_pylda $5.;
	informat create_pcoa $5.;
	*informat time ANYDTDTE.;
	*informat end_time ANYDTDTE.;
	informat time YMDDTTM26.6;
	informat end_time YMDDTTM26.6;

   format _index $84. ;
   format time_key $1024.;
   format type $53. ;
   format num_workers 3.;
   format batch_size best32. ;
   format num_documents 3.;
   format text_sha256 $64. ;
   format text_md5 $32. ;
   format convergence best32. ;
   format perplexity best32. ;
   format coherence best32. ;
   format topics best32. ;
   format alpha_str $18. ;
   format n_alpha best32. ;
   format beta_str $18. ;
   format n_beta best32. ;
    format passes best32. ;
    format iterations best32. ;
    format update_every best32. ;
    format eval_every best32. ;
    format chunksize best32. ;
    format random_state best32. ;
    format per_word_topics $5. ;
    format top_words $2048. ;
	format create_pylda $5.;
	format create_pcoa $5.;
	format time datetime26.6;
	format end_time datetime26.6;
 input
	     _index $
	    time_key $
	    type $
	    num_workers 
	    batch_size 
	    num_documents 
	    text_sha256 $
	    text_md5 $
	    convergence 
	    perplexity 
	    coherence 
	    topics 
	    alpha_str $
	    n_alpha 
	    beta_str $
	    n_beta 
	     passes 
	     iterations 
	     update_every
	     eval_every
	     chunksize
	     random_state
	     per_word_topics $
	     top_words $
		 create_pylda $
		 create_pcoa $
		 time 
		 end_time 
	;
 ;
 index = _n_;
 if _ERROR_ then call symputx('_EFIERR_',1);  /* set ERROR detection macro variable */
run;


/* Define the structure of the current_increment_data */
data current_increment_data;
    *length type $20 text $2000 text_sha256 text_md5 convergence 8 perplexity 8 coherence 8 topics 8 alpha_str $100 n_alpha 8 beta_str $100 n_beta 8 passes 8 iterations 8 update_every 8 eval_every 8 chunksize 8 random_state 8 per_word_topics $20  time datetime.;
	set metadata;
run;


proc sql;
	select *  from metadata;
quit;


proc sql;
    connect to postgres (user=postgres password='admin' server='127.0.0.1' port=5432   database=lda_manager_test);

	execute( DROP TABLE metadata) by postgres;
   execute (
      CREATE TABLE metadata (
         i TEXT,
         t VARCHAR(100),
         batch_size INT,
         text VARCHAR(255),
         text_sha256 VARCHAR(64),
         text_md5 VARCHAR(32),
         convergence FLOAT,
         perplexity FLOAT,
         coherence FLOAT,
         topics INT,
         alpha_str VARCHAR(100),
         n_alpha FLOAT,
         beta_str VARCHAR(100),
         n_beta FLOAT,
         passes FLOAT,
         iterations INT,
         update_every INT,
         eval_every INT,
     	 chunksize INT ,
      	 random_state INT ,
      	 per_word_topics BOOLEAN ,
      	 top_words VARCHAR(1000) ,
      	 time TIMESTAMP
      )
   ) by postgres;
	disconnect from postgres;
quit; 


data metadata_fixed;
	set metadata(rename=(coherence=_coherence perplexity=per_word_bound));
	label _perplexity_ = "_perplexity_ = 2^(-per_word_bound)"
		  per_word_bound = "NLL ''Negative Log Likelihood''";
	if _coherence = compress(_coherence,,'ka') > 0 then coherence=input('0.00000',best32.);
	else coherence=input(_coherence, best32.);
	_perplexity_ = 2**(-1*per_word_bound);
	drop _coherence;
run;

proc sort data=metadata_fixed out=metadata_sorted; by index text_sha256; run;
ods graphics on / width=1000PX;
ods html close;
ods pdf file = "C:\_harvester\data\lda-models\2020-2024\2020-2024-univariate.pdf";
title c=red "2020 - 2024 Univariate of Train Data";
ods proclabel="Univariate of Train Data";
proc univariate data=metadata_sorted(where=(type='train')) plots; by text_sha256; var coherence per_word_bound _perplexity_; *histogram coherence perplexity _perplexity; run;

title "Train Frequency Table";
ods proclabel="Train Frequency Table";
proc freq data=metadata_sorted(where=(type="train")); by text_sha256; tables alpha_str*beta_str; tables n_alpha*n_beta; run;

title c=red "2020 - 2024 Univariate of Eval Data";
ods proclabel="Univariate of Eval Data";
proc univariate data=metadata_sorted(where=(type='eval')) plots; by text_sha256;var coherence per_word_bound _perplexity_; *histogram coherence perplexity _perplexity; run;
title "Eval Frequency Table";
ods proclabel="Eval Frequency Table";
proc freq data=metadata_sorted(where=(type="eval")); by text_sha256; tables alpha_str*beta_str;  tables n_alpha*n_beta; run;

ods pdf close;
ods html;

ods graphics on / width=750PX;
proc sort data=metadata_fixed out=metadata_sorted; by topics; run;
proc sgplot data=metadata_sorted(where=(coherence>.8));
series x=topics y=coherence / group=type;
*series x=index y=per_word_bound / x2axis;
run;
proc sgplot data=metadata_sorted(where=(topics=110));
series x=index y=coherence / group=type;
*series x=index y=per_word_bound / x2axis;
run;

proc sql noprint;
	create table explore as
	select a.type, a.text_sha256 as left_sha, c.text_sha256 as right_sha, a.alpha_str, a.beta_str, c.topics, count(*) as sha_count label = "Distinct SHA256", sum(type_alpha_beta_count) as model_sum, type_alpha_beta_count label="Distinct Model(topic, alpha, beta)"
	from metadata as a,
		 (select distinct text_sha256, count(*) as sha_count label="Distinct SHA256 Count" from metadata group by text_sha256 having count(*) > 0) as b,
		 (select distinct type, topics, text_sha256, n_alpha, n_beta, count(*) as type_alpha_beta_count from metadata group by type, topics, n_alpha, n_beta having count(*) > 0) as c
	where /*a.text_sha256 = b.text_sha256 and*/
		  /*a.text_sha256 = '10723ca6bec4953f3f700d705cdc78fbac404922f2fbca8f1db3d8bf8370d4bb' and */
		  a.type = c.type and
		  a.text_sha256 = c.text_sha256 and
		  a.n_beta = c.n_beta and
		  a.n_alpha = c.n_alpha and
		  b.text_sha256=c.text_sha256
	group by a.type, a.text_sha256
	order by a.text_sha256, topics, alpha_str, beta_str, a.type;
/*
	select text into :text separated by '; '
	from metadata
	group by topics, alpha_str, beta_str
	having count(*) > 0
	order by alpha_str, beta_str;
	*%put NOTE: %bquote("&text");
*/
quit;

proc sort data=metadata_fixed out=explore_freq; by type text_sha256; run;
proc freq data=explore_freq;
	by type text_sha256;
	tables topics*alpha_str*beta_str;
run;

%macro get_zip();
	%do i=1 %to %sysfunc(countw(&text, ';'));
	%let zipfile = %scan(&text,&i);
		filename inzip ZIP "''%bquote(&zipfile)''";
		data open_csv;
			set explore;

			infile inzip lrecl=32767 recfm=F length=length unbuf;
			length text_out $ 32767;

			text_out=_infile_;
		run;
	%end;
%mend;

%get_zip()

