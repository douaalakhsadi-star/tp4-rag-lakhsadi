package ma.emsi.lakhsadi.tp4.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;

import java.util.List;

public class RagNaif {

    public static void main(String[] args) {
        String cle = System.getenv("GEMINI_KEY");

        ChatModel modele = GoogleGenAiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.7)
                .build();

        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        System.out.println(
                "Caractères extraits du PDF : " + document.text().length()
        );

        DocumentSplitter splitter = DocumentSplitters.recursive(1_000, 100);
        List<TextSegment> segments = splitter.split(document);

        System.out.println("Nombre de passages : " + segments.size());
    }
}