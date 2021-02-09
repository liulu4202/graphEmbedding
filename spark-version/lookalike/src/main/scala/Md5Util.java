import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.UnsupportedEncodingException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Calendar;

public class Md5Util {

    private static final Logger LOG = LoggerFactory.getLogger(Md5Util.class);

    /**
     * MD5摘要方式
     */
    private static final String MESSAGE_DIGEST_MD5 = "MD5";
    /**
     * 字符集UTF-8
     */
    private static final String CHARSET_UTF8 = "UTF-8";
    /**
     * 字符0
     */
    private static final String STRING_ZERO = "0";

    private Md5Util() {

    }

    /**
     * 
     * 将input进行md5摘要 32位
     *
     * @param input
     * @return
     */
    public static String evaluate(String input) {

        StringBuilder hexValue = new StringBuilder();

        try {

            MessageDigest md5 = MessageDigest.getInstance(MESSAGE_DIGEST_MD5);

            byte[] md5Bytes = md5.digest(input.getBytes(CHARSET_UTF8));

            for (int i = 0; i < md5Bytes.length; i++) {
                int val = ((int) md5Bytes[i]) & 0xff;
                if (val < 16)
                    hexValue.append(STRING_ZERO);
                hexValue.append(Integer.toHexString(val));
            }

        } catch (NoSuchAlgorithmException e) {
            LOG.error("MD5Util evaluate fail:{}", e);
        } catch (UnsupportedEncodingException e) {
            LOG.error("MD5Util evaluate encoding fail:{}", e);
        }

        return hexValue.toString();
    }

    /**
     *
     * 功能描述: 加密-32位小写 <br>
     * 〈功能详细描述〉
     *
     * @param encryptStr
     * @return
     * @see [相关类/方法](可选)
     * @since [产品/模块版本](可选)
     */
    public static String encrypt32(String encryptStr) {
        MessageDigest md5;
        try {
            md5 = MessageDigest.getInstance(MESSAGE_DIGEST_MD5);
            byte[] md5Bytes = md5.digest(encryptStr.getBytes(CHARSET_UTF8));
            StringBuffer hexValue = new StringBuffer();
            for (int i = 0; i < md5Bytes.length; i++) {
                int val = ((int) md5Bytes[i]) & 0xff;
                if (val < 16)
                    hexValue.append(STRING_ZERO);
                hexValue.append(Integer.toHexString(val));
            }
            encryptStr = hexValue.toString();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return encryptStr;
    }

    /**
     *
     * 功能描述: 加密-16位小写 <br>
     * 〈功能详细描述〉
     *
     * @param encryptStr
     * @return String
     * @see [相关类/方法](可选)
     * @since [产品/模块版本](可选)
     */
    public static String encrypt16(String encryptStr) {
        return encrypt32(encryptStr).substring(8, 24);
    }

    /**
     * 获取当前系统所处分钟
     *
     * @return
     */
    public static int getMinute() {
        Calendar calendar = Calendar.getInstance();
        calendar = Calendar.getInstance();
        calendar.getTime();
        return  calendar.get(Calendar.HOUR)*60+calendar.get(Calendar.MINUTE);
    }
}
