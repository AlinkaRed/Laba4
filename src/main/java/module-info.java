module org.example.processoremulatorfx {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
  requires jdk.jfr;

  opens org.example.processoremulatorfx to javafx.fxml;
    exports org.example.processoremulatorfx;
    exports org.example.processoremulatorfx.Controllers;
    opens org.example.processoremulatorfx.Controllers to javafx.fxml;
  exports org.example.processoremulatorfx.Models;
  opens org.example.processoremulatorfx.Models to javafx.fxml;
  exports org.example.processoremulatorfx.BModels;
  opens org.example.processoremulatorfx.BModels to javafx.fxml;
}