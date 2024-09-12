
data WORK.METADATA    ;
%let _EFIERR_ = 0; /* set the ERROR detection macro variable */
infile 'C:\_harvester\data\lda-models\2010s_html\metadata\metadata.csv' delimiter = ';' flowover DSD lrecl=32767 firstobs=2 TERMSTR=cr;
   informat VAR1 $84. ;
   informat type $53. ;
   informat batch_size best32. ;
   informat text $87. ;
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
    informat per_word_topics $4. ;
    informat top_words $78. ;
    format VAR1 $84. ;
    format type $53. ;
    format batch_size best12. ;
    format text $87. ;
    format text_sha256 $64. ;
    format text_md5 $32. ;
    format convergence best12. ;
    format perplexity best12. ;
    format coherence best12. ;
    format topics best12. ;
    format alpha_str $18. ;
    format n_alpha best12. ;
    format beta_str $18. ;
    format n_beta best12. ;
    format passes best12. ;
    format iterations best12. ;
    format update_every best12. ;
    format eval_every best12. ;
    format chunksize best12. ;
    format random_state best12. ;
    format per_word_topics $4. ;
    format top_words $78. ;
 input
             VAR1  $
             type  $
             batch_size
             text  $
             text_sha256  $
             text_md5  $
             convergence
             perplexity
             coherence
             topics
             alpha_str  $
             n_alpha
             beta_str  $
             n_beta
             passes
             iterations
             update_every
             eval_every
             chunksize
             random_state
             per_word_topics  $
             top_words  $
 ;
 if _ERROR_ then call symputx('_EFIERR_',1);  /* set ERROR detection macro variable */
run;


filename wordcld "C:\_harvester\data\lda-models\2010s_html\visuals\wordcloud-09082024.txt";
data topwords;
	length all_words found $ 32767;
	set metadata;
	file wordcld;

	if _n_ = 1 then do;
		regex = prxparse("/['](([A-Z])+(_)?([A-Z]))[']/i");
		retain regex;
		call missing(all_words);
	end;
	
	start = 1;
	stop = length(top_words);
	position=0;

	call prxnext(regex, start, stop, top_words, position, length);
	do while (position > 0);
	         found = substr(top_words, position, length);
	         *put found= position= length=;
			 all_words = strip(all_words) || ' ' || strip(dequote(found));
			 call prxnext(regex, start, stop, top_words, position, length);
	 end;
	 put all_words;
	 call missing(all_words);
run;
