import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class ChessPieceValidator {

    public static class ChessPieceMapper extends Mapper<Object, Text, Text, Text> {

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            context.write(new Text("all_pieces"), value);  // Pass the input line as key-value
        }
    }

    public static class ChessPieceReducer extends Reducer<Text, Text, Text, Text> {
        private static final String[] whitePieces = {"King", "Queen", "Rook", "Bishop", "Knight", "Pawn"};
        private static final String[] blackPieces = {"King", "Queen", "Rook", "Bishop", "Knight", "Pawn"};

        private static final Map<String, String[]> whiteStartingPositions = new HashMap<>();
        private static final Map<String, String[]> blackStartingPositions = new HashMap<>();

        private Map<String, Integer> whitePieceCount = new HashMap<>();
        private Map<String, Integer> blackPieceCount = new HashMap<>();

        private Set<String> validWhitePositions = new HashSet<>();
        private Set<String> validBlackPositions = new HashSet<>();
        private Set<String> conflictingPositions = new HashSet<>();
        private Set<String> invalidPositions = new HashSet<>();

        @Override
        protected void setup(Context context) {
            whiteStartingPositions.put("Rook", new String[]{"A1", "H1"});
            whiteStartingPositions.put("Knight", new String[]{"B1", "G1"});
            whiteStartingPositions.put("Bishop", new String[]{"C1", "F1"});
            whiteStartingPositions.put("Queen", new String[]{"D1"});
            whiteStartingPositions.put("King", new String[]{"E1"});
            whiteStartingPositions.put("Pawn", new String[]{"A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2"});

            blackStartingPositions.put("Rook", new String[]{"A8", "H8"});
            blackStartingPositions.put("Knight", new String[]{"B8", "G8"});
            blackStartingPositions.put("Bishop", new String[]{"C8", "F8"});
            blackStartingPositions.put("Queen", new String[]{"D8"});
            blackStartingPositions.put("King", new String[]{"E8"});
            blackStartingPositions.put("Pawn", new String[]{"A7", "B7", "C7", "D7", "E7", "F7", "G7", "H7"});
        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Map<String, Set<String>> whitePiecePositions = new HashMap<>();
            Map<String, Set<String>> blackPiecePositions = new HashMap<>();
            Map<String, Integer> whitePieceCounts = new HashMap<>();
            Map<String, Integer> blackPieceCounts = new HashMap<>();

            // Initialize piece counts for both White and Black
            initializePieceCounts(whitePieceCounts, whitePieces);
            initializePieceCounts(blackPieceCounts, blackPieces);

            // Process each line and update piece counts and positions
            for (Text value : values) {
                String[] parts = value.toString().split(" ");
                if (parts.length != 3) continue;  // Skip malformed lines

                String color = parts[0];
                String piece = parts[1];
                String position = parts[2];

                if (!isValidPosition(position)) {
                    invalidPositions.add(color + " " + piece + " at " + position);
                    continue; // Skip further processing for invalid positions
                }

                // Process piece positions
                if (color.equals("White")) {
                    processPiece(whitePieceCounts, whitePiecePositions, piece, position);
                } else if (color.equals("Black")) {
                    processPiece(blackPieceCounts, blackPiecePositions, piece, position);
                }
            }

            // Identify conflicting positions
            identifyConflicts(whitePiecePositions, conflictingPositions, "White");
            identifyConflicts(blackPiecePositions, conflictingPositions, "Black");

            // Write output for White Pieces
            context.write(new Text("WHITE PIECES:"), null);
            writeValidPositions(context, "White", whiteStartingPositions, whitePiecePositions, whitePieceCounts);
            writeConflictingPositions(context, "White");
            writeInvalidPositions(context, "White");
            writeMissingAndExtraPieces(context, "White", whitePieceCounts);

            // Write output for Black Pieces
            context.write(new Text("BLACK PIECES:"), null);
            writeValidPositions(context, "Black", blackStartingPositions, blackPiecePositions, blackPieceCounts);
            writeConflictingPositions(context, "Black");
            writeInvalidPositions(context, "Black");
            writeMissingAndExtraPieces(context, "Black", blackPieceCounts);
        }

        private void processPiece(Map<String, Integer> pieceCounts, Map<String, Set<String>> piecePositions, String piece, String position) {
            pieceCounts.put(piece, pieceCounts.get(piece) - 1);  // Reduce the count for that piece type
            // Use a set to track pieces at the same position
            piecePositions.computeIfAbsent(position, k -> new HashSet<>()).add(piece);
        }

        private void identifyConflicts(Map<String, Set<String>> piecePositions, Set<String> conflicts, String color) {
            for (Map.Entry<String, Set<String>> entry : piecePositions.entrySet()) {
                String position = entry.getKey();
                Set<String> piecesAtPosition = entry.getValue();
                if (piecesAtPosition.size() > 1) {
                    // Report conflicting pieces
                    conflicts.addAll(piecesAtPosition.stream()
                        .map(piece -> color + " " + piece + " in [" + position + "] at the same time")
                        .collect(Collectors.toList()));
                }

                // Check for duplicates of the same piece in the same position
                Map<String, Integer> pieceCountMap = new HashMap<>();
                for (String piece : piecesAtPosition) {
                    pieceCountMap.put(piece, pieceCountMap.getOrDefault(piece, 0) + 1);
                }
                for (Map.Entry<String, Integer> entryCount : pieceCountMap.entrySet()) {
                    if (entryCount.getValue() > 1) {
                        conflicts.add(entryCount.getKey() + " in [" + position + "] at the same time");
                    }
                }
            }
        }

        private void writeValidPositions(Context context, String color, Map<String, String[]> startingPositions,
                                         Map<String, Set<String>> piecePositions, Map<String, Integer> pieceCounts) throws IOException, InterruptedException {
            context.write(new Text("Valid starting positions:"), null);
            for (String position : piecePositions.keySet()) {
                Set<String> piecesAtPosition = piecePositions.get(position);
                if (piecesAtPosition.size() == 1) {  // Only write valid positions if no conflict
                    String piece = piecesAtPosition.iterator().next();
                    if (isInStartingPosition(startingPositions, piece, position) && !invalidPositions.contains(piece + " at " + position)) {
                        context.write(null, new Text(color + " " + piece + " in [" + position + "]"));
                    }
                }
            }

            context.write(new Text("Valid non-starting positions:"), null);
            for (String position : piecePositions.keySet()) {
                Set<String> piecesAtPosition = piecePositions.get(position);
                if (piecesAtPosition.size() == 1) {  // Only write valid non-starting positions if no conflict
                    String piece = piecesAtPosition.iterator().next();
                    if (!isInStartingPosition(startingPositions, piece, position) && !invalidPositions.contains(piece + " at " + position)) {
                        context.write(null, new Text(color + " " + piece + " in [" + position + "]"));
                    }
                }
            }
        }

        private void writeConflictingPositions(Context context, String color) throws IOException, InterruptedException {
            Set<String> conflictsForColor = new HashSet<>();
            for (String conflict : conflictingPositions) {
                if (conflict.startsWith(color)) {
                    conflictsForColor.add(conflict);
                }
            }
            if (!conflictsForColor.isEmpty()) {
                context.write(new Text("Conflicting positions:"), new Text(conflictsForColor.toString()));
            } else {
                context.write(new Text("Conflicting positions:"), new Text("null"));
            }
        }

        private void writeInvalidPositions(Context context, String color) throws IOException, InterruptedException {
            Set<String> invalidForColor = new HashSet<>();
            for (String invalid : invalidPositions) {
                if (invalid.startsWith(color)) {
                    invalidForColor.add(invalid);
                }
            }
            if (!invalidForColor.isEmpty()) {
                context.write(new Text("Invalid positions (not on board):"), new Text(invalidForColor.toString()));
            } else {
                context.write(new Text("Invalid positions (not on board):"), new Text("null"));
            }
        }

        private void writeMissingAndExtraPieces(Context context, String color, Map<String, Integer> pieceCounts) throws IOException, InterruptedException {
            context.write(new Text("Missing pieces (regardless of positional validity):"), null);
            for (String piece : pieceCounts.keySet()) {
                int count = pieceCounts.get(piece);
                if (count > 0) {
                    context.write(null, new Text(count + " missing " + color + " " + piece));
                }
            }

            context.write(new Text("Extra pieces (regardless of positional validity):"), null);
            for (String piece : pieceCounts.keySet()) {
                int count = pieceCounts.get(piece);
                if (count < 0) {
                    context.write(null, new Text((-count) + " extra " + color + " " + piece));
                }
            }
        }

        private boolean isInStartingPosition(Map<String, String[]> startingPositions, String piece, String position) {
            if (startingPositions.containsKey(piece)) {
                for (String startPos : startingPositions.get(piece)) {
                    if (startPos.equals(position)) {
                        return true;
                    }
                }
            }
            return false;
        }

        private boolean isValidPosition(String position) {
            return position.matches("[A-H][1-8]");
        }

        private void initializePieceCounts(Map<String, Integer> pieceCounts, String[] pieces) {
            for (String piece : pieces) {
                int count = piece.equals("Pawn") ? 8 : (piece.equals("Rook") || piece.equals("Knight") || piece.equals("Bishop")) ? 2 : 1;
                pieceCounts.put(piece, count);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "chess piece validator");
        job.setJarByClass(ChessPieceValidator.class);
        job.setMapperClass(ChessPieceMapper.class);
        job.setReducerClass(ChessPieceReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
