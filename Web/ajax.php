<?php
    $arr_file_types = ['image/png', 'image/gif', 'image/jpg', 'image/jpeg'];
    
    if (!(in_array($_FILES['file']['type'], $arr_file_types))) {
        echo "false";
        return;
    }
    
    if (!file_exists('uploads')) {
        mkdir('uploads', 0777);
    }
    
    move_uploaded_file($_FILES['file']['tmp_name'],  $_FILES['file']['name']);
    
    //echo "File uploaded successfully.";

    exec("python Main.py -i".$_FILES['file']['name']." > output.txt");

    $myfile = fopen("output.txt", "r") or die("Unable to open file!");
    $res =  fgets($myfile);
    fclose($myfile);
    $res = substr($res,14,4);
    //echo "<".$res.">";
    $i = (int)$res;


    if($i==3333)echo "അ	MALAYALAM LETTER A";
    elseif($i==3334)echo "ആ	MALAYALAM LETTER AA";
    elseif($i==3335)echo "ഇ	MALAYALAM LETTER I";
    elseif($i==3336)echo "ഈ	MALAYALAM LETTER II";
    elseif($i==3337)echo "ഉ	MALAYALAM LETTER U";
    elseif($i==3338)echo "ഊ	MALAYALAM LETTER UU";
    elseif($i==3339)echo "ഋ	MALAYALAM LETTER VOCALIC R";
    elseif($i==3340)echo "ഌ	MALAYALAM LETTER VOCALIC L";
    elseif($i==3342)echo "എ	MALAYALAM LETTER E";
    elseif($i==3343)echo "ഏ	MALAYALAM LETTER EE";
    elseif($i==3344)echo "ഐ	MALAYALAM LETTER AI";
    elseif($i==3346)echo "ഒ	MALAYALAM LETTER O";
    elseif($i==3347)echo "ഓ	MALAYALAM LETTER OO";
    elseif($i==3348)echo "ഔ	MALAYALAM LETTER AU";
    elseif($i==3349)echo "ക	MALAYALAM LETTER KA";
    elseif($i==3350)echo "ഖ	MALAYALAM LETTER KHA";
    elseif($i==3351)echo "ഗ	MALAYALAM LETTER GA";
    elseif($i==3352)echo "ഘ	MALAYALAM LETTER GHA";
    elseif($i==3353)echo "ങ	MALAYALAM LETTER NGA";
    elseif($i==3354)echo "ച	MALAYALAM LETTER CA";
    elseif($i==3355)echo "ഛ	MALAYALAM LETTER CHA";
    elseif($i==3356)echo "ജ	MALAYALAM LETTER JA";
    elseif($i==3357)echo "ഝ	MALAYALAM LETTER JHA";
    elseif($i==3358)echo "ഞ	MALAYALAM LETTER NYA";
    elseif($i==3359)echo "ട	MALAYALAM LETTER TTA";
    elseif($i==3360)echo "ഠ	MALAYALAM LETTER TTHA";
    elseif($i==3361)echo "ഡ	MALAYALAM LETTER DDA";
    elseif($i==3362)echo "ഢ	MALAYALAM LETTER DDHA";
    elseif($i==3363)echo "ണ	MALAYALAM LETTER NNA";
    elseif($i==3364)echo "ത	MALAYALAM LETTER TA";
    elseif($i==3365)echo "ഥ	MALAYALAM LETTER THA";
    elseif($i==3366)echo "ദ	MALAYALAM LETTER DA";
    elseif($i==3367)echo "ധ	MALAYALAM LETTER DHA";
    elseif($i==3368)echo "ന	MALAYALAM LETTER NA";
    elseif($i==3369)echo "ഩ	MALAYALAM LETTER NNNA";
    elseif($i==3370)echo "പ	MALAYALAM LETTER PA";
    elseif($i==3371)echo "ഫ	MALAYALAM LETTER PHA";
    elseif($i==3372)echo "ബ	MALAYALAM LETTER BA";
    elseif($i==3373)echo "ഭ	MALAYALAM LETTER BHA";
    elseif($i==3374)echo "മ	MALAYALAM LETTER MA";
    elseif($i==3375)echo "യ	MALAYALAM LETTER YA";
    elseif($i==3376)echo "ര	MALAYALAM LETTER RA";
    elseif($i==3377)echo "റ	MALAYALAM LETTER RRA";
    elseif($i==3378)echo "ല	MALAYALAM LETTER LA";
    elseif($i==3379)echo "ള	MALAYALAM LETTER LLA";
    elseif($i==3380)echo "ഴ	MALAYALAM LETTER LLLA";
    elseif($i==3381)echo "വ	MALAYALAM LETTER VA";
    elseif($i==3382)echo "ശ	MALAYALAM LETTER SHA";
    elseif($i==3383)echo "ഷ	MALAYALAM LETTER SSA";
    elseif($i==3384)echo "സ	MALAYALAM LETTER SA";
    elseif($i==3385)echo "ഹ	MALAYALAM LETTER HA";
    elseif($i==3386)echo "ഺ	MALAYALAM LETTER TTTA";
    elseif($i==3430)echo "൦	MALAYALAM DIGIT ZERO";
    elseif($i==3431)echo "൧	MALAYALAM DIGIT ONE";
    elseif($i==3432)echo "൨	MALAYALAM DIGIT TWO";
    elseif($i==3433)echo "൩	MALAYALAM DIGIT THREE";
    elseif($i==3434)echo "൪	MALAYALAM DIGIT FOUR";
    elseif($i==3435)echo "൫	MALAYALAM DIGIT FIVE";
    elseif($i==3436)echo "൬	MALAYALAM DIGIT SIX";
    elseif($i==3437)echo "൭	MALAYALAM DIGIT SEVEN";
    elseif($i==3438)echo "൮	MALAYALAM DIGIT EIGHT";
    elseif($i==3439)echo "൯	MALAYALAM DIGIT NINE";
    elseif($i==3440)echo "൰	MALAYALAM NUMBER TEN";
    elseif($i==3441)echo "൱	MALAYALAM NUMBER ONE HUNDRED";
    elseif($i==3442)echo "൲	MALAYALAM NUMBER ONE THOUSAND";
    elseif($i==3443)echo "൳	MALAYALAM FRACTION ONE QUARTER";
    elseif($i==3444)echo "൴	MALAYALAM FRACTION ONE HALF";
    elseif($i==3445)echo "൵	MALAYALAM FRACTION THREE QUARTERS";
    elseif($i==3449)echo "൹	MALAYALAM DATE MARK";
    elseif($i==3450)echo "ൺ	MALAYALAM LETTER CHILLU NN";
    elseif($i==3451)echo "ൻ	MALAYALAM LETTER CHILLU N";
    elseif($i==3452)echo "ർ	MALAYALAM LETTER CHILLU RR";
    elseif($i==3453)echo "ൽ	MALAYALAM LETTER CHILLU L";
    elseif($i==3454)echo "ൾ	MALAYALAM LETTER CHILLU LL";
    elseif($i==3455)echo "ൿ	MALAYALAM LETTER CHILLU K";
    elseif($i==3461)echo "අ	SINHALA LETTER AYANNA";
    elseif($i==3462)echo "ආ	SINHALA LETTER AAYANNA";
    elseif($i==3463)echo "ඇ	SINHALA LETTER AEYANNA";
    elseif($i==3464)echo "ඈ	SINHALA LETTER AEEYANNA";
    elseif($i==3465)echo "ඉ	SINHALA LETTER IYANNA";
    elseif($i==3466)echo "ඊ	SINHALA LETTER IIYANNA";
    elseif($i==3467)echo "උ	SINHALA LETTER UYANNA";
    elseif($i==3468)echo "ඌ	SINHALA LETTER UUYANNA";
    elseif($i==3469)echo "ඍ	SINHALA LETTER IRUYANNA";
    elseif($i==3470)echo "ඎ	SINHALA LETTER IRUUYANNA";
    elseif($i==3471)echo "ඏ	SINHALA LETTER ILUYANNA";
    elseif($i==3472)echo "ඐ	SINHALA LETTER ILUUYANNA";
    elseif($i==3473)echo "එ	SINHALA LETTER EYANNA";
    elseif($i==3474)echo "ඒ	SINHALA LETTER EEYANNA";
    elseif($i==3475)echo "ඓ	SINHALA LETTER AIYANNA";
    elseif($i==3476)echo "ඔ	SINHALA LETTER OYANNA";
    elseif($i==3477)echo "ඕ	SINHALA LETTER OOYANNA";
    elseif($i==3478)echo "ඖ	SINHALA LETTER AUYANNA";



    /*if($i==3333)echo "അ";
    elseif($i==3334)echo "ആ";
    elseif($i==3335)echo "ഇ";
    elseif($i==3336)echo "ഈ";
    elseif($i==3337)echo "ഉ";
    elseif($i==3338)echo "ഊ";
    elseif($i==3339)echo "ഋ";
    elseif($i==3340)echo "ഌ";
    elseif($i==3342)echo "എ";
    elseif($i==3343)echo "ഏ";
    elseif($i==3344)echo "ഐ";	
    elseif($i==3346)echo "ഒ";
    elseif($i==3347)echo "ഓ";
    elseif($i==3348)echo "ഔ";
    elseif($i==3349)echo "ക";
    elseif($i==3350)echo "ഖ";
    elseif($i==3351)echo "ഗ";
    elseif($i==3352)echo "ഘ";
    elseif($i==3353)echo "ങ";
    elseif($i==3354)echo "ച";
    elseif($i==3355)echo "ഛ";
    elseif($i==3356)echo "ജ";
    elseif($i==3357)echo "ഝ";
    elseif($i==3358)echo "ഞ";
    elseif($i==3359)echo "ട";
    elseif($i==3360)echo "ഠ";
    elseif($i==3361)echo "ഡ";
    elseif($i==3362)echo "ഢ";
    elseif($i==3363)echo "ണ";
    elseif($i==3364)echo "ത";
    elseif($i==3365)echo "ഥ";
    elseif($i==3366)echo "ദ";
    elseif($i==3367)echo "ധ";
    elseif($i==3368)echo "ന";
    elseif($i==3369)echo "ഩ";
    elseif($i==3370)echo "പ";
    elseif($i==3371)echo "ഫ";
    elseif($i==3372)echo "ബ";
    elseif($i==3373)echo "ഭ";
    elseif($i==3374)echo "മ";
    elseif($i==3375)echo "യ";
    elseif($i==3376)echo "ര";
    elseif($i==3377)echo "റ";
    elseif($i==3378)echo "ല";
    elseif($i==3379)echo "ള";
    elseif($i==3380)echo "ഴ";
    elseif($i==3381)echo "വ";
    elseif($i==3382)echo "ശ";
    elseif($i==3383)echo "ഷ";
    elseif($i==3384)echo "സ";
    elseif($i==3385)echo "ഹ";
    elseif($i==3386)echo "ഺ";*/
?>