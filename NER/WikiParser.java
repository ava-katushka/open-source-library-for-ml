import edu.jhu.nlp.wikipedia.PageCallbackHandler;
import edu.jhu.nlp.wikipedia.WikiPage;
import edu.jhu.nlp.wikipedia.WikiXMLParser;
import edu.jhu.nlp.wikipedia.WikiXMLParserFactory;

import java.io.*;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Вычленение ссылок для каждой статьи википедии в отдельный файл
 */
//
public class WikiParser {
    static int i = 0;
    static HashMap<String, HashSet<String>> dirStructure = new HashMap<>();


    public static void main(String[] args) throws Exception {
        File path = new File (args[0]); //
        File folders[] = path.listFiles();
        HashSet<String> articlesSet = new HashSet<>();
        File folderLink = new File ("wikiLinks");
        folderLink.mkdirs();
        for (File f: folders) {
            for (File article: f.listFiles()) {
                articlesSet.add(article.getName());
            }
            folderLink = new File("wikiLinks/" + f.getName());
            folderLink.mkdirs();
            dirStructure.put(f.getName(), articlesSet);
            articlesSet = new HashSet<>();
        }
        WikiXMLParser wxsp = WikiXMLParserFactory.getSAXParser("enwiki-pages-articles.xml.bz2");
        try {
            wxsp.setPageCallback(new PageCallbackHandler() {
                public void process(WikiPage page) {
                    ++i;
                    String nameWithoutSlash = page.getTitle().replace('/', '_').split("[\n]+")[0];
                    String nameOfFolder = "different";
                    if (dirStructure.containsKey(nameWithoutSlash.substring(0, 1))) {
                        nameOfFolder = nameWithoutSlash.substring(0, 1);
                    }
                    HashSet<String> articlesInThisFolder = dirStructure.get(nameOfFolder);
                    if (articlesInThisFolder != null) {
                        if (articlesInThisFolder.contains(nameWithoutSlash)) {
                            articlesInThisFolder.remove(nameWithoutSlash);
                            try {
                                Writer writer = new BufferedWriter(new OutputStreamWriter(
                                        new FileOutputStream("wikiLinks/" + nameOfFolder + "/" + nameWithoutSlash), "utf-8"));

                                for (String link :  page.getLinks()) {
                                    writer.write(link + "\n");
                                }
                                writer.close();
                            } catch (Exception e) {
                            }
                        }
                    }

                }
            });
            wxsp.parse();

        } catch(Exception e) {
            e.printStackTrace();
        }

    
    }
}
