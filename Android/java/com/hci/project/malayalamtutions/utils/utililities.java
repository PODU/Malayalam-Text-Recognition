package com.hci.project.malayalamtutions.utils;

import java.util.Random;

public class utililities {

    public static int[] classes = {3333,3334,3335,3337,3342,3343,3346,3349,3350,3351,3352,3353,33354,3355,3356,3357,3358,3359,3360,3361,3362,3363,3364,3365,3366,3367,3368,3370,3371,3372,3373,3374,3375,3376,3377,3378,3379,3380,3381,3382,3383,3384,3385,3450,3451,3452,3453,3454};

    public static String[] chars = {"അ","ആ","ഇ","ഉ","എ","ഏ","ഒ","ക","ഖ","ഗ","ഘ","ങ","ച","ഛ","ജ","ഝ","ഞ","ട","ഠ","ഡ","ഢ","ണ","ത","ഥ","ദ","ധ","ന","പ","ഫ","ബ","ഭ","മ","യ","ര","റ","ല","ള","ഴ","വ","ശ","ഷ","സ","ഹ","ൺ","ൻ","ർ","ൽ","ൾ"};

    public static int getIndexOfClass(int n){
        for(int i=0;i<classes.length;i++){
            if(classes[i]==n){
                return i;
            }
        }
        return -1;
    }

    public static int getClassFromIndex(int n){
        if(n<classes.length){
            return classes[n];
        }else{
            return -1;
        }
    }

    public static int getIndexOfChar(String s){
        for(int i=0;i<chars.length;i++){
            if(chars[i].equals(s)){
                return i;
            }
        }
        return -1;
    }

    public static String getCharsFromIndex(int n){
        if(n<chars.length){
            return chars[n];
        }else{
            return null;
        }
    }

    public static int getRandomClasses(){
        Random r = new Random();
        int n = r.nextInt(classes.length);
        return classes[n];
    }


}
