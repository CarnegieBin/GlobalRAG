_2HOP_COMPARISON_QA_PROMPT = """
Examples:
=============================
Wikipedia Title: Kurram Garhi
Kurram Garhi is a small village located near the city of Bannu, which is the part of Khyber Pakhtunkhwa province of Pakistan. Its population is approximately 35000. Barren hills are near this village. This village is on the border of Kurram Agency. Other nearby villages are Peppal, Surwangi and Amandi Kala.

Wikipedia Title: 2001–02 UEFA Champions League second group stage
Eight winners and eight runners- up from the first group stage were drawn into four groups of four teams, each containing two group winners and two runners- up. Teams from the same country or from the same first round group could not be drawn together. The top two teams in each group advanced to the quarter- finals.

Wikipedia Title: Satellite tournament
A satellite tournament is either a minor tournament or event on a competitive sporting tour or one of a group of such tournaments that form a series played in the same country or region.\n\nWikipedia Title: Trojkrsti\nTrojkrsti is a village in Municipality of Prilep, Republic of Macedonia.\n\nWikipedia Title: Telephone numbers in Ascension Island\nCountry Code:+ 247< br> International Call Prefix: 00 Ascension Island does not share the same country code( +290) with the rest of St Helena.

Question: Are both Kurram Garhi and Trojkrsti located in the same country?
Answer:
```json
{{"Response": "No"}}
```
=============================
Wikipedia Title: Cassandra's Dream (album)
Cassandra's Dream is the soundtrack to the 2008 Woody Allen film" Cassandra's Dream" and features an original orchestral score by Philip Glass. The soundtrack was released on January 8, 2008.

Wikipedia Title: Inspire (La'Mule album)
Inspire is the first full- length studio album by visual kei rock group La'Mule. The album was released through Bandai Music Entertainment on December 2, 1998. Two songs from the album, Mind Control and Usagi no Tsumi, had been released earlier that year as a double single.

Wikipedia Title: What's Inside
What's Inside is the fourteenth studio album by British singer- songwriter Joan Armatrading. The album was written, arranged and produced by Armatrading, co-produced by David Tickle and recorded at the A&M Recording Studios in Hollywood. The strings were recorded at Abbey Road Studios, London, with the Kronos Quartet's contribution recorded at The Plant Recording Studios, Sausalito, California and The Memphis Horns recorded at Kiva Recording Studio, Memphis, Tennessee. The album was released in 1995 by RCA and was Armatrading's only album for the label. She had left A&M in 1992 after an eighteen- year association with the company.

Wikipedia Title: Drama of the Ages
Drama of the Ages is the third album by metal band Jacobs Dream and was released in 2005. It is the first Jacobs Dream album to feature singer Chaz Bond on vocals. The album contains an untitled hidden song on the final track which is an instrumental version of Pachelbel's Canon in D.

Question: Which album was released earlier, What'S Inside or Cassandra'S Dream (Album)?
Answer:
```json
{{"Response": "What's Inside"}}
```
=============================
Wikipedia Title: John Allen (Oxford University cricketer)
John Aubrey Allen( born 19 July 1974 in Windsor, New South Wales) is an Australian- born first- class cricketer who played for Oxford University Cricket Club. Both his first- class games were for Oxford University, one of which was a varsity match.

Wikipedia Title: Hartley Lobban
Hartley W Lobban (9 May 1926 – 15 October 2004) was a Jamaican-born first-class cricketer who played 17 matches for Worcestershire in the early 1950s.

Wikipedia Title: Martin Hodge
Martin Hodge( born 4 February 1959 in Southport, Lancashire) is an English former professional footballer. He played as a goalkeeper for Plymouth Argyle, Everton, Sheffield Wednesday, Leicester City, Hartlepool United and Rochdale. His career lasted from 1977 to 1996 during which time he played 602 league and cup matches.

Wikipedia Title: Ivania Martinich
Ivania Martinich Soriano( born 25 July 1995) is a Chilean tennis player. Martinich has a career high WTA singles ranking of 692 achieved on 4 November 2013. She also has a career high WTA doubles ranking of 826, achieved on 31 March 2014. Playing for Chile in Fed Cup, Martinich has a win- loss record of 1–1.

Question: Who was born first out of Martin Hodge and Ivania Martinich?
Answer:
```json
{{"Response": "Martin Hodge"}}
```
=============================

NOTE: Always respond with the JSON object.
Now it's your turn!
"""

_2HOP_BRIDGE_QA_PROMPT = """
Examples:
=============================
Wikipedia Title: Hypocrite (film)
Hypocrite (Spanish: Hipócrita..!) is a 1949 Mexican thriller film directed by Miguel Morayta and starring Antonio Badú, Leticia Palma, Carmen Molina and Luis Beristáin. The film included the song "Hipócrita". The film's sets were designed by Francisco Marco Chillet.

Wikipedia Title: Bill Smith (footballer, born 1897)
William Thomas Smith( born 9 April 1897, date of death unknown) was an English professional footballer.

Wikipedia Title: Harry Wainwright (footballer)
Harry Wainwright( born 1899; date of death unknown) was an English footballer.

Wikipedia Title: Miguel Morayta
Miguel Morayta (15 August 1907 – 19 June 2013) was a Spanish film director and screenwriter. He directed 74 films between 1944 and 1978. At the outbreak of the Spanish Civil War, Morayta was a Spanish artillery officer, who joined the Republican side. After Francisco Franco's victory, he left Spain for France and Africa, finally arriving in Mexico in 1941, where he started his career. He was living in Mexico when he died aged 105.

Question: When did the director of film Hypocrite (Film) die?
Answer:
```json
{{"Response": "19 June 2013"}}
```
=============================
Wikipedia Title: Ryu Hye-young
Ryu Hye-young (born March 28, 1991) is a South Korean actress and model. She is best known for her role as Sung Bo-ra in the hit drama series "Reply 1988" (2015-2016).

Wikipedia Title: Reply 1988
Reply 1988 () is a South Korean television series starring Lee Hye-ri, Park Bo-gum, Go Kyung-pyo, Ryu Jun-yeol and Lee Dong-hwi. Set in the year 1988, it revolves around five friends and their families living in the same neighborhood of Ssangmun-dong, Dobong District, Northern Seoul. It aired every Friday and Saturday from November 6, 2015, to January 16, 2016, on tvN at 7:50 (KST) for 20 episodes.

Wikipedia Title: Lawyers (TV series)
Lawyers ( is a 2005 South Korean television series starring Jung Hye-young, Kim Sang-kyung, Kim Sung-soo and Han Go-eun. It aired on MBC from July 4 to August 23, 2005 on Monday and Tuesday at 21:55 for 16 episodes.

Question: How many episodes were in the South Korean television series in which Ryu Hye-young played Bo-ra?
Answer:
```json
{{"Response": "20"}}
```
=============================
Wikipedia Title: ISO/TC 68
ISO/TC 68 is a technical committee formed within the International Organization for Standardization (ISO), of Geneva, Switzerland, tasked with developing and maintaining international standards covering the areas of banking, securities, and other financial services. As the standards organization under ISO responsible for the development of all international financial services standards, ISO/TC 68 plays a key role in the development and adoption of new technologies in the banking, brokerage and insurance industries. Many of its current work projects involve developing ecommerce standards such as better online security for financial transactions, XML standards for financial transactions and standards to reduce the cost and delays of international financial transactions. The membership of ISO/TC 68, consists of more than 30 organizations assigned by participating national standards bodies plus additional international standards development organizations that work collaboratively toward global financial services standards development.

Wikipedia Title: ISO 21500
ISO 21500:2012, Guidance on Project Management, is an international standard developed by the International Organization for Standardization, or ISO starting in 2007 and released in 2012. It was intended to provide generic guidance, explain core principles and what constitutes good practice in project management. The ISO technical committee dealing with project management, ISO/PC 236 was held by the American National Standards Institute (ANSI) which had approved four standards that used PMI materials. one of which was ANSI/PMI 99-001-2008, A Guide to the Project Management Body of Knowledge - 4th Edition (PMI BoK® Guide - 4th Edition) (revision and re-designation of ANSI/PMI 99-001-2004): 11/20/2008.

Wikipedia Title: ISO 3166-2:BM
ISO 3166-2:BM is the entry for Bermuda in ISO 3166-2, part of the ISO 3166 standard published by the International Organization for Standardization (ISO), which defines codes for the names of the principal subdivisions (e.g., provinces or states) of all countries coded in ISO 3166-1.

Wikipedia Title: ISO 3166-1
ISO 3166-1 is part of the ISO 3166 standard published by the International Organization for Standardization (ISO), and defines codes for the names of countries, dependent territories, and special areas of geographical interest. The official name of the standard is "Codes for the representation of names of countries and their subdivisions – Part 1: Country codes". It defines three sets of country codes:

Question: What is the headquarters for the organization who sets the standards for ISO 21500?
Answer:
```json
{{"Response": "Geneva"}}
```
=============================

NOTE: Always respond with the JSON object.
Now it's your turn!
"""

_2HOP_INFERENCE_QA_PROMPT = """
Examples:
=============================
Wikipedia Title: Hypocrite (film)
Hypocrite (Spanish: Hipócrita..!) is a 1949 Mexican thriller film directed by Miguel Morayta and starring Antonio Badú, Leticia Palma, Carmen Molina and Luis Beristáin. The film included the song "Hipócrita". The film's sets were designed by Francisco Marco Chillet.

Wikipedia Title: Bill Smith (footballer, born 1897)
William Thomas Smith( born 9 April 1897, date of death unknown) was an English professional footballer.

Wikipedia Title: Harry Wainwright (footballer)
Harry Wainwright( born 1899; date of death unknown) was an English footballer.

Wikipedia Title: Miguel Morayta
Miguel Morayta (15 August 1907 – 19 June 2013) was a Spanish film director and screenwriter. He directed 74 films between 1944 and 1978. At the outbreak of the Spanish Civil War, Morayta was a Spanish artillery officer, who joined the Republican side. After Francisco Franco's victory, he left Spain for France and Africa, finally arriving in Mexico in 1941, where he started his career. He was living in Mexico when he died aged 105.

Question: When did the director of film Hypocrite (Film) die?
Answer:
```json
{{"Response": "19 June 2013"}}
```
=============================
Wikipedia Title: Rudra Shah
Rudra Shah (?–1673) was the king of the Gorkha Kingdom in the Indian subcontinent, present-day Nepal. He was the father of Prithvipati Shah.

Wikipedia Title: Constance Anne Herschel
Constance Anne Herschel( 1855- 1939), later known as Lady Lubbock, was a scientist and mathematician. Herschel held the post of resident lecturer in natural sciences and mathematics at Girton College, Cambridge. She was the child of Sir John Frederick William Herschel, and the grandchild of William Herschel. She wrote a family history of the famous scientific dynasty by compiling family sources,' The Herschel Chronicle'. She married Sir Neville Lubbock.

Wikipedia Title: Krishna Shah (Nepalese royal)
Krishna Shah (?–1661) was the king of the Gorkha Kingdom in the Indian subcontinent, present-day Nepal. He was the father of Rudra Shah.

Question: Who is the grandchild of Krishna Shah (Nepalese Royal)?
Answer:
```json
{{"Response": "Prithvipati Shah"}}
```
=============================
Wikipedia Title: Peter Burroughs
Peter Burroughs( born 27 January 1947) is a British television and film actor, the director of Willow Management. He is the father- in- law of actor and TV presenter Warwick Davis.

Wikipedia Title: Boraqchin (wife of Ögedei)
Boraqchin was the first and eldest wife of Ögedei Khan. They had no surviving children. The earliest known Sino-Mongolian inscription, from 1240, mentions a "Yeke Qadun" or "Great empress". Some scholars have identified this figure with Boraqchin, while others argue that the inscription refers to Ögedei Khan's second wife, Töregene Khatun.

Wikipedia Title: Ögedei Khan
Ögedei (also Ogodei; , Mongolian: "ÖgedeiÖgüdei"; ; c.1186– 11 December 1241) was the third son of Genghis Khan and second Great Khan of the Mongol Empire, succeeding his father. He continued the expansion of the empire that his father had begun, and was a world figure when the Mongol Empire reached its farthest extent west and south during the Mongol invasions of Europe and East Asia. Like all of Genghis' primary sons, he participated extensively in conquests in China, Iran, and Central Asia.

Wikipedia Title: John Vernou Bouvier III
John Vernou" Black Jack" Bouvier III( May 19, 1891 – August 3, 1957) was an American Wall Street stockbroker and socialite. He was the father of First Lady Jacqueline Kennedy Onassis and of socialite Lee Radziwill, and was the father- in- law of John F. Kennedy.

Question: Who is Boraqchin (Wife Of Ögedei)'s father-in-law?
Answer:
```json
{{"Response": "Genghis Khan"}}
```
=============================

NOTE: Always respond with the JSON object.
Now it's your turn!
"""

_3HOP_BRIDGE_QA_PROMPT = """
Examples:
=============================
Wikipedia Title: Morris Lake (Nova Scotia)
Morris Lake is a long shallow lake in Nova Scotia's Halifax Regional Municipality bordering the communities of Dartmouth, Shearwater and Cole Harbour.

Wikipedia Title: Mexico–United States border
The total length of the continental border is 3,201 kilometers (1,989 mi). From the Gulf of Mexico, it follows the course of the Rio Grande (Río Bravo del Norte) to the border crossing at Ciudad Juárez, Chihuahua and El Paso, Texas. Westward from El Paso -- Juárez, it crosses vast tracts of the Chihuahuan and Sonoran deserts to the Colorado River Delta and San Diego -- Tijuana, before reaching the Pacific Ocean.

Wikipedia Title: Fort Salonga, New York
Fort Salonga is a hamlet and census-designated place (CDP) in Suffolk County, New York on the North Shore of Long Island. At the 2010 census, the CDP population was 10,008. The name evolved from the Revolutionary War-era British Fort Salonga, or Fort Slongo, (named after one of the fort’s architects) once located near the border of the towns of Huntington and Smithtown, overlooking Long Island Sound.

Wikipedia Title: Mexico–United States border
Among the U.S. states, Texas has the longest stretch of the border with Mexico, while California has the shortest. Among the states in Mexico, Chihuahua has the longest border with the United States, while Nuevo León has the shortest.

Wikipedia Title: Finding Dory
One year later, Dory is living with Marlin and Nemo on their reef. One day, Dory has a flashback and remembers that she has parents. She decides to look for them, but her memory problem is an obstacle. She eventually remembers that they lived at the Jewel of Morro Bay across the ocean in California, thanks to Nemo mentioning its name.

Question: How long is the US border with the country that borders the state where Finding Dory takes place?
Answer:
```json
{{"Response": "1,989 mi"}}
```
=============================
Wikipedia Title: Magic Tour Highlights
Magic Tour Highlights is an EP by Bruce Springsteen and the E Street Band, which consists of four live audio tracks and their accompanying videos, and was released for digital download on July 15, 2008. The performances were recorded during the 2008 Magic Tour, and feature guest musicians, as well as Danny Federici's last performance with the group.

Wikipedia Title: The Antidote (Ronny Jordan album)
The Antidote is the debut album by English jazz guitarist Ronny Jordan, that was released by Island Records in 1992.

Wikipedia Title: The Crush Tour (album)
The Crush Tour is a third concert video by American band Bon Jovi from the European leg of their Crush Tour. It was recorded on August 30, 2000 at Zurich, Switzerland. It was directed by Anthony Bongiovi. It was released on DVD in 2001.

Wikipedia Title: Bounce (Bon Jovi album)
Bounce is the eighth studio album by American rock band Bon Jovi, released on October 8, 2002 through Island Records. Produced by Luke Ebbin, Jon Bon Jovi and Richie Sambora, the album was recorded at Sanctuary II Studio in New Jersey.

Wikipedia Title: 30th Anniversary Tour: Live
30th Anniversary Tour: Live is the fourth live album by George Thorogood and the Destroyers. It was recorded on May 4, 2004 at the Royal Concert Hall in Nottingham, England, and on October 19, 2004 on the Eagle Records label. The performance was also released on DVD, and as a CD/DVD collectors' edition.

Wikipedia Title: Juggernaut (Hunters & Collectors album)
Juggernaut is the ninth and final studio album by Australian rock band, Hunters & Collectors. The album, recorded in 1997, was co-produced by the group with Kalju Tonuma and Mark Opitz. It was released on 26 January 1998 on Mushroom's White Label. With its release, Hunters & Collectors announced they would disband after the Say Goodbye Tour – they gave their final performances in late March 1998. The album peaked at No. 36 on the ARIA Albums Chart and No. 48 on the New Zealand Albums Chart.

Question: What is the genre of the record label of the band that performed on the Crush Tour?
Answer:
```json
{{"Response": "jazz"}}
```
=============================
Wikipedia Title: Compton, California
Compton is a city in southern Los Angeles County, California, United States, situated south of downtown Los Angeles. Compton is one of the oldest cities in the county and on May 11, 1888, was the eighth city to incorporate. As of the 2010 United States Census, the city had a total population of 96,456. It is known as the "Hub City" due to its geographic centrality in Los Angeles County. Neighborhoods in Compton include Sunny Cove, Leland, Downtown Compton, and Richland Farms. The city is generally a working class city with some middle-class neighborhoods, and is home to a relatively young population, at an average 25 years of age, compared to the American median age of 38 (based on 2018 data).

Wikipedia Title: Cherokee City, Arkansas
Cherokee City is an unincorporated census-designated place in Benton County, Arkansas, United States. As of the 2010 census, its population is 72. It is the location of (or is the nearest community to) Coon Creek Bridge, which is located on Cty Rd. 24 and is listed on the National Register of Historic Places. The community was named for the Cherokee Indians, since the Trail of Tears crossed the landscape when the Cherokee migrated west to Indian territory, now Oklahoma in the late 1830s. The town is about 5 miles east of Oklahoma and 4 miles south of the Missouri state line.

Wikipedia Title: MC Eiht
Aaron Tyler (born May 22, 1971), better known by his stage name MC Eiht, is an American rapper and actor. Many of his songs are based on his life in Compton. His stage name was partly inspired by the numeral in KRS-One's name. He chose Eiht for its links to "hood culture", including Olde English 800 (8 Ball) and .38 caliber firearms. He is the "de facto" leader of West Coast hip hop group Compton's Most Wanted, which also included fellow Compton-based rappers Boom Bam, Tha Chill, DJ Mike T, DJ Slip and Ant Capone. He is also known for his role as A-Wax in the 1993 film "Menace II Society".

Wikipedia Title: Smoke in tha City
Smoke in tha City is the ninth studio album by American rapper MC Eiht, released September 14, 2004 on Factor House Records. It was produced by Black C-Zer and Quincy Miller. The album featuring guest performances by West Coast heavy-weights: RBX, Spice 1, Kokane, Jayo Felony and Daz Dillinger.

Question: In which county was the birthplace of the Smoke in tha City performer?
Answer:
```json
{{"Response": "Los Angeles County"}}
```
=============================

NOTE: Always respond with the JSON object.
Now it's your turn!
"""

_4HOP_BRIDGE_COMPARISON_QA_PROMPT =  """
Examples:
=============================
Wikipedia Title: Ian Barry (director)
Ian Barry is an Australian director of film and TV.

Wikipedia Title: Coolie No. 1 (1995 film)
Coolie No. 1 is a 1995 Indian Hindi- language comedy film directed by David Dhawan, and written by Rumi Jaffery and Kader Khan. The film stars Govinda, Karisma Kapoor, Shakti Kapoor, Kader Khan and Sadashiv Amrapurkar, with music by Anand- Milind. This movie was one of the first successful movies of Karisma Kapoor. Actor Govinda received the Star Screen Award Special Jury Award for his role in this movie as' performer of the decade'. Over the years, the movie has become a classic in Hindi film history and is now considered a cult film. The movie is a remake of the 1993 Tamil film Chinna Mapillai.

Wikipedia Title: The Sensational Trial
The Sensational Trial is a 1923 German silent film directed by Karl Freund and starring Erich Kaiser- Titz, Käthe Haack and Heinrich Schroth. The film's sets were designed by the art director Heinrich Richter.

Wikipedia Title: David Dhawan
David Dhawan( born Rajinder Dhawan on 16 August 1955) is an Indian film director who works in Hindi films. He is the father of Bollywood actor Varun Dhawan and director Rohit Dhawan. He is best known for directing several successful films, including" Swarg"( 1990)," Shola Aur Shabnam"( 1992)," Saajan Chale Sasural"( 1996)," Judwaa"( 1997)," Bade Miyan Chote Miyan"( 1998)," Dulhan Hum Le Jayenge"( 2000)," Mujhse Shaadi Karogi"( 2004)," Partner"( 2007)," Chashme Baddoor"( 2013)," Main Tera Hero"( 2014) and" Judwaa 2"( 2017). The 1993 action thriller" Aankhen" and 1999 comedy" Biwi No.1" earned him the Filmfare Award for Best Director nominations.

Wikipedia Title: Karl Freund
Karl W. Freund, A.S.C. (January 16, 1890 – May 3, 1969) was a German cinematographer and film director best known for photographing "Metropolis" (1927), "Dracula" (1931), and television's "I Love Lucy" (1951-1957). Freund was an innovator in the field of cinematography and is credited with the invention of the unchained camera technique.

Question: Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?
Answer:
```json
{{"Response": "no"}}
```
=============================
Wikipedia Title: Maurice Campbell
Maurice Campbell( November 28, 1919 – July 4, 2014) was a Canadian curler from Trois- Rivières, Quebec. Campbell was born November 28, 1919 in Saint- Hyacinthe, Quebec. Educated at the University of Montreal, he joined the Royal Canadian Army Medical Corps in 1943 and completed his medical degree in 1945 and was subsequently posted in Halifax, Nova Scotia. He was a specialist in rheumatology and internal medicine and practiced in Cap- de- la- Madeleine, Quebec. Campbell played in the 1958 Macdonald Brier, playing lead for the Quebec team, skipped by Bob Lahaie. The team finished 9th, with a 3- 7 record. He was President of the Quebec Curling Association for the 1963- 64 season. He served as president of the Canadian Curling Association for the 1970- 71 season and was named to the Canadian Curling Hall of Fame. Campbell died in Trois- Rivières on July 4, 2014.

Wikipedia Title: Joy Mukherjee
Joy Mukherjee( 24 February 1939 – 9 March 2012) was an Indian film actor and director. He was titled as the' Heart throb of the 60's'.

Wikipedia Title: Nariman Irani
Nariman A. Irani(?- 10 December 1977) was a Bollywood cinematographer and film producer. He is most known for producing" Don"( 1978) made under his banner Nariman Films and for his work as cinematographer in Chhailla Babu( 1977). He died in an accident, even before" Don" was even completed, eventually the film was a big hit and led to the Don film franchise. As a cinematographer he is known for his films like" TalashSaraswatichandra" and" Phool Aur PattharRoti Kapada Aur Makaan" and Chhailla Babu. He won the 16th National Film Awards for Best Cinematography( B& W) for" Saraswatichandra"( 1968)., he also won the Filmfare Award for Best Cinematographer in the same year. The cinematographer Nariman Irani, while working Chhailla Babu decided to borrow most of the plot of Chhailla Babu and shared a modified story idea to Chandra Barot, who made the new modified story as the film Don( 1978). While" Don" was still under production, he was badly hurt, when after a sudden cloudburst in November 1977, a wall fell on him while he was preparing to take a shot for another film at Rajkamal Kalamandir studios, Mumbai during the making of Manojj Kumar's Kranti.

Wikipedia Title: Two Weeks with Pay
Two Weeks with Pay is a lost 1921 American silent comedy romance film starring Bebe Daniels and directed by Maurice Campbell.

Wikipedia Title: Chhailla Babu
Chhailla Babu ( Alternate name: Chhaila Babu)( English language: Cool Guy) is a 1977 Bollywood suspense thriller film. It was written and produced by Shomu Mukherjee and directed by Joy Mukherjee. Rajesh Khanna plays the title role of Chhailla Babu. It stars Zeenat Aman, Asrani, Om Shivpuri and Ranjeet. The film's music is by Laxmikant Pyarelal. It had a collection of 4 crores in 1977. The film became a superhit. This was the only film directed by Joy to be successful at the box office. and the success of" Chhailla Babu" gave a boost to the career of Khanna, who went through a bad phase between 1976 and 1978 at the box office. This film seems to be very heavily inspired by the Cary Grant/ Audrey Hepburn super hit film CHARADE. The cinematographer in this film, Nariman Irani, while working on this film decided to borrow most of the plot of" Chhailla Babu" and shared a modified story idea to Chandra Barot, who made the new modified story as the film" Don"( 1978).

Question: Which film has the director born first, Two Weeks With Pay or Chhailla Babu?
Answer:
```json
{{"Response": "Two Weeks With Pay"}}
```
=============================
Wikipedia Title: Gladiators 7
Gladiators 7 is a 1962 film directed by Pedro Lazaga. The film has several elements from Akira Kurosawa's film" The Seven Samurai".

Wikipedia Title: Arthur Penn
Arthur Hiller Penn (September 27, 1922 – September 28, 2010) was an American director and producer of film, television and theater. Closely associated with the American New Wave, Penn directed critically acclaimed films throughout the 1960s such as the drama "The Chase" (1966), the biographical crime film "Bonnie and Clyde" (1967) and the comedy "Alice's Restaurant" (1969). He also got attention for his revisionist Western "Little Big Man" (1970). By the mid-1970s his films were received with much less enthusiasm. In the 1990s he returned to stage and television direction and production, including an executive producer role for the crime series "Law & Order". By his death in 2010, he had been nominated for three Academy Awards for Best Director, a BAFTA, a Golden Globe, two Emmys, and two Directors Guild of America Awards. He was the recipient of several honorary accolades, including a Honorary Golden Bear, a Tony Award, and an Akira Kurosawa Award from the San Francisco International Film Festival.

Wikipedia Title: Franz Josef Gottlieb
Franz Josef Gottlieb (1 November 1930 – 23 July 2006) was an Austrian film director and screenwriter. He directed 60 films between 1959 and 2005. He also directed the children's series "Ravioli" in 1983; it aired on ZDF in 1984. He was born in Semmering, Austria and died in Verden an der Aller, Germany of a brain tumor.

Wikipedia Title: The Avenger (1962 film)
The Avenger is a 1962 film directed by Giorgio Venturini.

Wikipedia Title: The Miracle Worker (1962 film)
The Miracle Worker is a 1962 American biographical film about Anne Sullivan, blind tutor to Helen Keller, directed by Arthur Penn. The screenplay by William Gibson is based on his 1959 play of the same title, which originated as a 1957 broadcast of the television anthology series" Playhouse 90". Gibson's secondary source material was" The Story of My Life", the 1902 autobiography of Helen Keller. The film went on to be an instant critical success and a moderate commercial success. The film was nominated for five Academy Awards, including Best Director for Arthur Penn, and won two awards, Best Actress for Anne Bancroft and Best Supporting Actress for Patty Duke. " The Miracle Worker" also holds a perfect 100% score from the movie critics site Rotten Tomatoes.

Wikipedia Title: When the Mad Aunts Arrive
When the Mad Aunts Arrive is a 1970 West German musical comedy film directed by Franz Josef Gottlieb and starring Rudi Carrell, Ilja Richter and Chris Roberts. In one of a group of films during the era inspired by the cross- dressing plot of" Charley's Aunt", two men end up at a Carinthian hotel resort disguised as females. This leads to confusion during their romantic pursuit of woman. The film was shot on location in Munich and the Wörthersee in Austria. The film's sets were designed by Eberhard Schröder.

Question: Which film has the director died earlier, When The Mad Aunts Arrive or The Miracle Worker (1962 Film)?
Answer:
```json
{{"Response": "When The Mad Aunts Arrive"}}
```
=============================

NOTE: Always respond with the JSON object.
Now it's your turn!
"""

_4HOP_BRIDGE_QA_PROMPT = """
Examples:
Wikipedia Title: World Series of Poker
The World Series of Poker (WSOP) is a series of poker tournaments held annually in Las Vegas and, since 2005, sponsored by Caesars Entertainment Corporation (known as Harrah's Entertainment until 2010). It dates its origins to 1970, when Benny Binion invited seven of the best - known poker players to the Horseshoe Casino for a single tournament, with a set start and stop time, and a winner determined by a secret ballot of the seven players.

Wikipedia Title: Gold Spike (property)
Gold Spike (formerly Gold Spike Hotel & Casino) is a bar, lounge, residential building, and former boutique 112 - room, seven floor hotel. It is connected with the Oasis at the Gold Spike, a 50 - room three floor hotel located in downtown Las Vegas. It was owned by entrepreneur Tony Hsieh and his Downtown Project, having bought it from The Siegel Group; and the casino was operated by Golden Gaming.

Wikipedia Title: 2003 World Series of Poker
The 2003 World Series of Poker (WSOP) was held at Binion's Horseshoe. There were 839 entrants to the main event. Each paid $10,000 to enter what was the largest poker tournament ever played in a brick and mortar casino at the time. Many entrants, including the overall winner Chris Moneymaker, won their seat in online poker tournaments. The 2003 Main Event was the first tournament to pay out at least $2,500,000 to the winner. Dan Harrington made the final table and looked to win his second Main Event championship, but fell short in third place.

Wikipedia Title: LinkExchange
It was founded in March 1996 by 23-year-old Harvard graduates Tony Hsieh (who later went on to invest in and become the CEO of Zappos) and Sanjay Madan. Ali Partovi later joined them as a third partner in August 1996. In November 1996, when the company consisted of about 10 people, it moved from Hsieh's and Madan's living room to an office in San Francisco. In May 1997, the company received US$3 million in funding from Sequoia Capital.

Wikipedia Title: Harvard University
Harvard's 2,400 professors, lecturers, and instructors instruct 7,200 undergraduates and 14,000 graduate students. The school color is crimson, which is also the name of the Harvard sports teams and the daily newspaper, The Harvard Crimson. The color was unofficially adopted (in preference to magenta) by an 1875 vote of the student body, although the association with some form of red can be traced back to 1858, when Charles William Eliot, a young graduate student who would later become Harvard's 21st and longest-serving president (1869\u20131909), bought red bandanas for his crew so they could more easily be distinguished by spectators at a regatta.

Wikipedia Title: Norman Chad
out of a tournament). He also utters various humorous phrases whenever he mentions a player's alma mater. The phrase is almost always, ""I believe they are the..."" finished with an unusual college nickname, such as Demon Deacons or Ragin' Cajuns. Since 2003, Chad has appeared on most of ESPN's poker broadcasts, including the World Series of Poker and The United States Poker Championship, among other events. Since becoming ESPN's poker analyst, Chad has also participated in 50 World Series of Poker events himself. In both 2009 and 2011, he finished in the money of the $1,500 Stud 8 or better

Wikipedia Title: Harvard University
Harvard University is a private Ivy League research university in Cambridge, Massachusetts, United States. Founded October 28, 1636, and named for its first benefactor, the Puritan clergyman John Harvard, it is the oldest institution of higher learning in the United States. Its influence, wealth, and rankings have made it one of the most prestigious universities in the world.

Wikipedia Title: Gold Spike
The Gold Spike, for many years, was known as an inexpensive downtown Las Vegas hotel with decent rooms, limited amenities, and a decent sized casino. On December 6, 2002, Jackie Gaughan agreed to sell the Gold Spike and three other casinos to Barrick Gaming. This sale, along with several other downtown Las Vegas hotel/casinos, was completed in 2004, for a combined total of about $82 million. Barrick Gaming Corp was in partnership with Tamares Group. After the purchase, management discontinued the table games and only offered slot machines. In a few years the property was offered for sale.

Question: What is the enrollment of undergraduates at the alma mater of the man who owns the gold spike in the home of the World Series of Poker?
Answer:
```json
{{"Response": "7,200"}}
```
=============================
Wikipedia Title: Culture of Aruba
The culture of Aruba, one of the many islands that make up the Caribbean, is an amalgamate of the various cultures that have occupied and lived on the island, including indigenous peoples of South America, descendants of African slaves, and Spanish and Dutch colonialists.

Wikipedia Title: Aruba
Aruba is geologically located in South-America, lying on the South-American continental shelf. Alongside Bonaire and Curac\u0327ao, Aruba forms a group referred to as the ABC islands. The Dutch Caribbean encompasses the ABC islands along with the other three substantial islands, the SSS islands. In contrast to much of the Caribbean, which experiences humid tropical climates, Aruba has a dry climate with an arid xeric landscape. The relatively warm and sunny weather persists throughout the year.

Wikipedia Title: Loures (parish)
Loures is a civil parish in the municipality of Loures, Portugal. It is an urban parish, part of the city of Loures. The population in 2011 was 26,769, in an area of 32.82\u00a0km\u00b2.

Wikipedia Title: Portuguese Empire
Although the royal family returned to Portugal in 1821, the interlude led to a growing desire for independence amongst Brazilians. In 1822, the son of Dom Jo\u00e3o VI, then prince - regent Dom Pedro I, proclaimed the independence of Brazil on September 7, 1822, and was crowned Emperor of the new Empire of Brazil. Unlike the Spanish colonies of South America, Brazil's independence was achieved without significant bloodshed.

Wikipedia Title: Supreme court
In Brazil, the Supreme Federal Tribunal (Supremo Tribunal Federal) is the highest court. It is both the constitutional court and the court of last resort in Brazilian law. It only reviews cases that may be unconstitutional or final habeas corpus pleads for criminal cases. It also judges, in original jurisdiction, cases involving members of congress, senators, ministers of state, members of the high courts and the President and Vice-President of the Republic. The Superior Court of Justice (Tribunal Superior de Justiça) reviews State and Federal Circuit courts decisions for civil law and criminal law cases, when dealing with federal law or conflicting rulings. The Superior Labour Tribunal (Tribunal Superior do Trabalho) reviews cases involving labour law. The Superior Electoral Tribunal (Tribunal Superior Eleitoral) is the court of last resort of electoral law, and also oversees general elections. The Superior Military Tribunal (Tribunal Superior Militar) is the highest court in matters of federal military law.

Wikipedia Title: The parish of Loures
The parish of Loures was first mentioned in 1118, and again in 1191, along with the parishes of Unhos and Sacavém.[2] At the time the region was mainly agricultural, and involved in small export trade, that included the cultivation of lettuce (a vegetable that continues to be cultivated). The first written document appeared in 1170, written by Afonso Henriques, who provided privileges and benefits to the Moors that lived on the outskirts of Lisbon in the sallayos (the fields to the north of Lisbon, at the time). In 1178, King Sancho I of Portugal having discovered that the Moors of Lisbon were allied to groups in Loures, and came to the territory with a Templar force to confront those peoples. Following the defeat of the Moors, the King transferred the region to the Knights Templar, who established the sect of warrior-monks. The extinction of Templars in 1311, opened the way for the expropriation of lands and property by the Pope.

Wikipedia Title: Minister Plenipotentiary of Aruba
Minister Plenipotentiary of Aruba The Minister Plenipotentiary of Aruba () represents the constituent country of Aruba in the Council of Ministers of the Kingdom of the Netherlands. The current Minister Plenipotentiary of Aruba is Guillfred Besaril. The Minister Plenipotentiary and his cabinet are seated in the Arubahuis (Aruba House) in The Hague. A significant difference between the Netherlands ministers and the Ministers Plenipotentiary is that the former ministers are accountable for their politics and policies to the Dutch parliament. The Ministers Plenipotentiary, however, are accountable to their national governments, which is the Estates of Aruba in case of Aruba. Therefore,

Wikipedia Title: Supreme court
In most legal jurisdictions, a supreme court, also known as a court of last resort, apex court, and high (or final) court of appeal, and court of final appeal, is the highest court within the hierarchy of courts. Broadly speaking, the decisions of a supreme court are binding on all other courts in a nation and are not subject to further review by any other court. Supreme courts typically function primarily as appellate courts, hearing appeals from decisions of lower trial courts, or from intermediate-level appellate courts. A supreme court can also, in certain circumstances, act as a court of original jurisdiction.

Question: What is the highest court in the country being the colonial holding in the continent having Aruba governed by the country having Loures?
Answer:
```json
{{"Response": "Supreme Federal Tribunal (Supremo Tribunal Federal)"}}
```
=============================
Wikipedia Title: Khabarovsk
Khabarovsk is served by the Khabarovsk Novy Airport with international flights to East Asia, Southeast Asia, European Russia, and Central Asia.

Wikipedia Title: Pacific National University
Pacific National University (PNU) is one of the largest universities in Khabarovsk Russia. It was established in 1958. Today the university trains over 21,000 in 54 majors.

Wikipedia Title: Iran
In 1941, Reza Shah was forced to abdicate in favor of his son, Mohammad Reza Pahlavi, and established the Persian Corridor, a massive supply route that would last until the end of the ongoing war. The presence of so many foreign troops in the nation also culminated in the Soviet-backed establishment of two puppet regimes in the nation; the Azerbaijan People's Government, and the Republic of Mahabad. As the Soviet Union refused to relinquish the occupied Iranian territory, the Iran crisis of 1946 was followed, which particularly resulted in the dissolution of both puppet states, and the withdrawal of the Soviets.

Wikipedia Title: Armenia
The Karabakh war ended after a Russian-brokered cease-fire was put in place in 1994. The war was a success for the Karabakh Armenian forces who managed to capture 16% of Azerbaijan's internationally recognised territory including Nagorno-Karabakh itself. Since then, Armenia and Azerbaijan have held peace talks, mediated by the Organisation for Security and Co-operation in Europe (OSCE). The status of Karabakh has yet to be determined. The economies of both countries have been hurt in the absence of a complete resolution and Armenia's borders with Turkey and Azerbaijan remain closed. By the time both Azerbaijan and Armenia had finally agreed to a ceasefire in 1994, an estimated 30,000 people had been killed and over a million had been displaced.

Wikipedia Title: Iran
On September 22, 1980, the Iraqi army invaded the Iranian Khuzestan, and the Iran–Iraq War began. Although the forces of Saddam Hussein made several early advances, by mid 1982, the Iranian forces successfully managed to drive the Iraqi army back into Iraq. In July 1982, with Iraq thrown on the defensive, Iran took the decision to invade Iraq and conducted countless offensives in a bid to conquer Iraqi territory and capture cities, such as Basra. The war continued until 1988, when the Iraqi army defeated the Iranian forces inside Iraq and pushed the remaining Iranian troops back across the border. Subsequently, Khomeini accepted a truce mediated by the UN. The total Iranian casualties in the war were estimated to be 123,220–160,000 KIA, 60,711 MIA, and 11,000–16,000 civilians killed.

Wikipedia Title: Iran
Another civil war ensued after the death of Karim Khan in 1779, out of which Aqa Mohammad Khan emerged, founding the Qajar Dynasty in 1794. In 1795, following the disobedience of the Georgian subjects and their alliance with the Russians, the Qajars captured Tblisi by the Battle of Krtsanisi, and drove the Russians out of the entire Caucasus, reestablishing a short-lived Iranian suzerainty over the region. The Russo-Persian wars of 1804–1813 and 1826–1828 resulted in large irrevocable territorial losses for Iran in the Caucasus, comprising all of Transcaucasia and Dagestan, which made part of the very concept of Iran for centuries, and thus substantial gains for the neighboring Russian Empire.

Wikipedia Title: United Nations Regional Groups
the African Group, with 54 member states the Asia - Pacific Group, with 53 member states the Eastern European Group, with 23 member states the Latin American and Caribbean Group (GRULAC), with 33 member states the Western European and Others Group (WEOG), with 28 member states, plus 1 member state (the United States) as an observer state.

Question: How many countries in Pacific National University's continent are recognized by the organization that mediated the truce ending the Iran-Iraq war?
Answer:
```json
{{"Response": "53"}}
```
=============================

NOTE: Always respond with the JSON object.
Now it's your turn!
"""

_4HOP_COMPARISON_QA_PROMPT = """
Examples:
Wikipedia Title: James A. Haley
James Andrew Haley( January 4, 1899 – August 6, 1981) was a U.S. Representative from Florida.

Wikipedia Title: William Goodsir
William Ernest Goodsir( 12 January 1902 – 14 July 1958) was a New Zealand- born businessman, politician and rugby coach in Fiji.

Question: Who lived longer, James A. Haley or William Goodsir?
Answer:
```json
{{"Response": "James A. Haley"}}
```
=============================
Wikipedia Title: Constance Keys
Constance Mabel Keys (30 October 1886 – 17 March 1964) was one of the most highly decorated nurses from Australia who served in World War I. She was mentioned twice in despatches, was awarded the Royal Red Cross, First Class and the Médaille des Epidémies.

Wikipedia Title: Anthony de Jasay
Anthony de Jasay( 15 October 1925 – 23 January 2019) was a Hungarian writer, economist, and philosopher.

Wikipedia Title: Constance Keys
Egypt, later travelling onto Britain and then to France. Keys was born in Mount Perry, a small town in the Wide Bay–Burnett region, the seventh child of Irish immigrant James Keys, a schoolteacher, and his wife Margaret. Trained at the Brisbane General Hospital as a nurse, she enlisted in September 1914 in the

Wikipedia Title: Anthony de Countie
Anthony de Countie Anthony de Countie, also called Anthony Conti, Anthony de Conti and Anthony de County, (died 1579), was a Renaissance composer and lutenist, active in the 16th century at the Tudor court in England, and one of the principal lutenists of the Elizabethan era. According to David Lasocki, Anthony de Countie may have been a Spanish musician of Jewish origin, but he is more likely to have been an Italian, because he had an Italian wife, Lucretia, and he lived with another Italian, Francis Jetto, between 1565 and 1571. Anthony de Countie was employed as lutenist at the

Question: Who lived longer, Constance Keys or Anthony De Jasay?
Answer:
```json
{{"Response": "Anthony De Jasay"}}
```
=============================
Wikipedia Title: Lon Myers
Laurence Eugene" Lon" Myers( February 16, 1858 – February 16, 1899) was an American sprinter and middle distance runner.

Wikipedia Title: Antonio Superchi
Antonio Superchi (11 January 1816 \u2013 5 July 1893) was an Italian operatic baritone who had an active international career from 1838\u20131858.

Wikipedia Title: Lon Myers
Lon Myers\"\nHe was in the first graduating class of Richmond High School. His father moved the family to Jersey City, New Jersey, in 1875 after he graduated high school, and then to New York City, where he became a bookkeeper. During his 21-year career, Myers held every American record for races 50 yards to one mile. He won 15 United States national championships, 10 Canadian national championships, and 3 British national championships. From 1880 to 1888, he held the world records in the 100-yard, 440-yard, and 880-yard races. Myers began running competitively in 1878, for the Knickerbocker Yacht Club. He then

Wikipedia Title: Franco Superchi
Franco Superchi Franco Superchi (born September 1, 1944 in Allumiere) is a retired Italian professional football player. He played for 13 seasons in the Serie A for ACF Fiorentina, Hellas Verona F.C. and A.S. Roma. For the first 3 seasons with ACF Fiorentina he was the third-choice goalkeeper and did not play any league games for the first two. In his fourth, 1968/69 season, Fiorentina won the Italian championship, and the impressive performances by the rookie were integral to the team's success. He played 7 more seasons for Fiorentina, never equalling the brilliance of the championship season, but always reliable.

Question: Who lived longer, Lon Myers or Antonio Superchi?
Answer:
```json
{{"Response": "Antonio Superchi"}}
```
=============================

NOTE: Always respond with the JSON object.
Now it's your turn!
"""



